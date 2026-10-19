"""Live integration test reproducing the exact README flow against live Neo4j and gateway.

Marked strict xfail because on HEAD (9f8cc3d), chunk_id is stored on :Document,
not :Chunk, causing retrieval to raise RetrievalError.
When M3 fixes identity, strict=True ensures this test fails until the xfail marker
is deliberately removed.
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

from graphrag_pipeline.config import (
    Neo4jSettings,
    model_settings_from_env,
)
from graphrag_pipeline.extraction.run_fixture import DEFAULT_FIXTURE, run_fixture
from graphrag_pipeline.query import custom_query
from graphrag_pipeline.retrieval import RetrievalError, RetrievalSettings, build_retriever, retrieve
from graphrag_pipeline.storage.neo4j_connection import Neo4jConnection
from tools.model_gateway import run_gateway

GATEWAY_PORT = 18080
GATEWAY_URL = f"http://127.0.0.1:{GATEWAY_PORT}/v1"


def _ensure_gateway_running() -> None:
    """Ensure the local model gateway is running in stub mode on 127.0.0.1:18080."""
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
    """Get Neo4j settings or fail/skip based on REQUIRE_LIVE."""
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
            pytest.skip(f"Neo4j is not reachable at {settings.uri}; skipping live integration test.")
    return settings


@pytest.mark.live
def test_readme_flow_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run fixture ingestion, create vector index, and verify end-to-end retrieval."""
    _ensure_gateway_running()
    settings = _get_neo4j_settings()

    monkeypatch.setenv("OPENROUTER_BASE_URL", GATEWAY_URL)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-stub-token")
    monkeypatch.setenv("OPENROUTER_LLM_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")
    monkeypatch.setenv("OPENROUTER_EMBEDDING_DIMENSIONS", "1536")
    monkeypatch.setenv("GRAPH_RAG_VECTOR_DIMENSIONS", "1536")
    monkeypatch.setenv("GRAPH_RAG_PROVIDER", "openrouter")

    # Ingest fixture
    asyncio.run(run_fixture(DEFAULT_FIXTURE))

    # Create vector index
    with Neo4jConnection(settings) as conn:
        conn.run_write(
            "CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS FOR (c:Chunk) ON c.embedding "
            "OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}"
        )
        conn.run_write("CALL db.awaitIndexes(120)")

    model_settings = model_settings_from_env()
    retrieval_settings = RetrievalSettings()

    with Neo4jConnection(settings) as conn:
        retriever = build_retriever(
            conn.connect(),
            settings.database,
            model_settings,
            retrieval_settings=retrieval_settings,
        )
        request = custom_query(
            question="What company did Aster Systems acquire in 2021?",
            trace_id="live-test-01",
        )
        result = retrieve(request, retriever, retrieval_settings=retrieval_settings)
        assert result.has_evidence
        assert len(result.evidence) >= 1
        assert result.evidence[0].paragraph_hash is not None
        assert result.evidence[0].chunk_id == f"musique:paragraph:{result.evidence[0].paragraph_hash}"
        assert "Aster Systems" in result.evidence[0].text


@pytest.mark.live
def test_rerun_one_record_yields_distinct_paragraphs_in_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rerun one real record (20 paragraphs) through ingestion, then retrieve top 5.

    Confirms deduplication + skip-if-done prevents duplicate copies so the top 5
    retrieval results hold 5 distinct paragraphs rather than duplicate copies.
    """
    from graphrag_pipeline.ingestion.load_musique_dataset import load_dataset

    _ensure_gateway_running()
    settings = _get_neo4j_settings()

    monkeypatch.setenv("OPENROUTER_BASE_URL", GATEWAY_URL)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-stub-token")
    monkeypatch.setenv("OPENROUTER_LLM_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")
    monkeypatch.setenv("OPENROUTER_EMBEDDING_DIMENSIONS", "1536")
    monkeypatch.setenv("GRAPH_RAG_VECTOR_DIMENSIONS", "1536")
    monkeypatch.setenv("GRAPH_RAG_PROVIDER", "openrouter")

    dataset_path = Path("dataset/musique/data/data/musique_ans_v1.0_dev.jsonl")
    if not dataset_path.exists():
        pytest.skip(f"MuSiQue dataset not found at {dataset_path}")

    # First run: ingest record 1 (20 paragraphs)
    summary1 = asyncio.run(
        load_dataset(
            dataset_path=dataset_path,
            start_record=1,
            max_records=1,
            source_split="dev",
            dry_run=False,
            show_progress=False,
        )
    )
    assert len(summary1.chunks_succeeded) == 20 or summary1.chunks_skipped == 20

    # Second run: rerun record 1 -> must skip all 20 ($0 spend)
    summary2 = asyncio.run(
        load_dataset(
            dataset_path=dataset_path,
            start_record=1,
            max_records=1,
            source_split="dev",
            dry_run=False,
            show_progress=False,
        )
    )
    assert summary2.chunks_skipped == 20

    # Ensure vector index is online
    with Neo4jConnection(settings) as conn:
        conn.run_write(
            "CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS FOR (c:Chunk) ON c.embedding "
            "OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}"
        )
        conn.run_write("CALL db.awaitIndexes(120)")

    model_settings = model_settings_from_env()
    retrieval_settings = RetrievalSettings(top_k=5)

    with Neo4jConnection(settings) as conn:
        retriever = build_retriever(
            conn.connect(),
            settings.database,
            model_settings,
            retrieval_settings=retrieval_settings,
        )
        request = custom_query(
            question="When was the company founded?",
            trace_id="rerun-test-01",
        )
        result = retrieve(request, retriever, retrieval_settings=retrieval_settings)
        assert len(result.evidence) == 5
        distinct_hashes = {chunk.paragraph_hash for chunk in result.evidence}
        assert len(distinct_hashes) == 5, f"Expected 5 distinct paragraphs, got {len(distinct_hashes)}"

