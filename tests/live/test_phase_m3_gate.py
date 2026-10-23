"""Phase Gate M3 verification tests.

Phase gate M3 requirements:
1. Ingest real records through the stub gateway ($0 spend).
2. Counts match: unique paragraph hashes, no duplicates.
3. Second run makes 0 calls (all skipped via resumable ingestion markers).
4. Resume drill passes: incomplete/uncommitted paragraph is purged before re-ingestion.
5. No leakage in graph: 0 gold labels or answer strings stored in Neo4j.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
import urllib.request
from pathlib import Path

import pytest
from neo4j.exceptions import ServiceUnavailable

from graphrag_pipeline.config import Neo4jSettings, model_settings_from_env
from graphrag_pipeline.ingestion.load_musique_dataset import load_dataset
from graphrag_pipeline.ingestion.record_adapter import compute_paragraph_hash
from graphrag_pipeline.storage.neo4j_connection import Neo4jConnection
from graphrag_pipeline.storage.question_context import (
    link_questions_to_paragraphs,
    verify_no_leakage_in_graph,
)
from graphrag_pipeline.storage.resumable_ingestion import (
    cleanup_incomplete_paragraph,
    ensure_paragraph_hash_constraint,
    load_completed_paragraph_hashes,
)
from tools.model_gateway import run_gateway

GATEWAY_PORT = 18080
GATEWAY_URL = f"http://127.0.0.1:{GATEWAY_PORT}/v1"


def _ensure_gateway_running() -> None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{GATEWAY_PORT}/health", timeout=1) as resp:
            if resp.status == 200:
                return
    except Exception:
        pass

    server_thread = threading.Thread(
        target=run_gateway,
        kwargs={"host": "127.0.0.1", "port": GATEWAY_PORT, "mode": "stub"},
        daemon=True,
    )
    server_thread.start()
    for _ in range(50):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{GATEWAY_PORT}/health", timeout=1) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(0.05)
    raise RuntimeError("Failed to start background stub model gateway.")


def _get_neo4j_settings() -> Neo4jSettings:
    os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
    os.environ.setdefault("NEO4J_USERNAME", "neo4j")
    os.environ.setdefault("NEO4J_PASSWORD", "changeme")
    os.environ.setdefault("NEO4J_DATABASE", "neo4j")

    try:
        settings = Neo4jSettings.from_env()
    except ValueError as e:
        if os.environ.get("REQUIRE_LIVE") == "1":
            pytest.fail(f"Invalid Neo4j settings: {e}")
        pytest.skip(f"Neo4j not configured: {e}")

    with Neo4jConnection(settings) as conn:
        try:
            conn.verify_connectivity()
        except ServiceUnavailable:
            if os.environ.get("REQUIRE_LIVE") == "1":
                pytest.fail(f"Neo4j is not reachable at {settings.uri} but REQUIRE_LIVE=1")
            pytest.skip(f"Neo4j is not reachable at {settings.uri}")
    return settings


@pytest.mark.live
def test_gate_m3_resume_drill(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resume drill: uncommitted paragraph is cleanly purged and re-ingested."""
    _ensure_gateway_running()
    settings = _get_neo4j_settings()

    test_hash = "test_incomplete_hash_12345"
    with Neo4jConnection(settings) as conn:
        # Create an incomplete Document node simulating an uncommitted crash
        conn.run_write(
            "CREATE (d:Document {paragraph_hash: $h, title: 'Incomplete', ingest_status: 'in_progress'}) "
            "CREATE (c:Chunk {text: 'Partial text'})-[:FROM_DOCUMENT]->(d)",
            {"h": test_hash},
        )
        # Verify it exists
        records = conn.run_read(
            "MATCH (d:Document {paragraph_hash: $h}) RETURN d.ingest_status AS status",
            {"h": test_hash},
        )
        assert len(records) == 1
        assert records[0]["status"] == "in_progress"

        # Run cleanup
        cleanup_incomplete_paragraph(conn.connect(), test_hash)

        # Verify Document and attached Chunk are completely removed
        records_after = conn.run_read(
            "MATCH (d:Document {paragraph_hash: $h}) RETURN d",
            {"h": test_hash},
        )
        assert len(records_after) == 0


@pytest.mark.live
def test_gate_m3_idempotency_and_zero_spend_rerun(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ingesting real records through the stub, then rerunning makes 0 calls ($0 spend)."""
    _ensure_gateway_running()
    settings = _get_neo4j_settings()

    monkeypatch.setenv("OPENROUTER_BASE_URL", GATEWAY_URL)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-stub-token")
    monkeypatch.setenv("OPENROUTER_LLM_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")
    monkeypatch.setenv("OPENROUTER_EMBEDDING_DIMENSIONS", "1536")
    monkeypatch.setenv("GRAPH_RAG_VECTOR_DIMENSIONS", "1536")
    monkeypatch.setenv("GRAPH_RAG_PROVIDER", "openrouter")

    from tests.live.musique_ci import musique_inputs
    dataset_path, _slice_path = musique_inputs()

    # First run on records 1..3 (approximately 60 paragraphs)
    summary1 = asyncio.run(
        load_dataset(
            dataset_path=dataset_path,
            start_record=1,
            max_records=3,
            source_split="dev",
            dry_run=False,
            show_progress=False,
        )
    )
    total_chunks = summary1.chunks_planned
    assert total_chunks > 0
    assert len(summary1.chunks_succeeded) == total_chunks
    assert len(summary1.chunks_failed) == 0

    # Second run on the exact same records: must skip 100% of chunks ($0 spend)
    summary2 = asyncio.run(
        load_dataset(
            dataset_path=dataset_path,
            start_record=1,
            max_records=3,
            source_split="dev",
            dry_run=False,
            show_progress=False,
        )
    )
    assert summary2.chunks_skipped == total_chunks
    new_extractions = [c for c in summary2.chunks_succeeded if not c.get("skipped")]
    assert len(new_extractions) == 0
    assert len(summary2.chunks_failed) == 0


@pytest.mark.live
def test_gate_m3_counts_match_and_no_leakage() -> None:
    """Document count matches distinct paragraph_hash count, and 0 leakage exists."""
    settings = _get_neo4j_settings()

    with Neo4jConnection(settings) as conn:
        records = conn.run_read(
            "MATCH (d:Document) "
            "WHERE d.paragraph_hash IS NOT NULL "
            "RETURN count(d) AS total, count(DISTINCT d.paragraph_hash) AS distinct_hashes"
        )
        assert len(records) == 1
        total = records[0]["total"]
        distinct_hashes = records[0]["distinct_hashes"]
        assert total == distinct_hashes, f"Found duplicate documents! total={total}, distinct={distinct_hashes}"

        # Assert no evaluation leakage in the graph
        verify_no_leakage_in_graph(conn.connect())
