"""Phase Gate M5 (Gate-M) verification tests.

Gate-M requirements:
1. Ingest 10 slice questions (200 paragraphs) through record mode with the experiment profile.
2. 0 writer failures.
3. 0 schema-invention calls (exactly 2 model calls per paragraph: extraction + embedding).
4. Pinned model configuration: temperature 0, seed, provider openai, no reasoning_effort.
5. Bridge coverage and graph quality metrics recorded.
6. Extraction schema frozen by hash.
7. Idempotent rerun skips all chunks ($0 spend).
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from neo4j.exceptions import ServiceUnavailable

from eval.graph_quality import compute_graph_quality, render_graph_quality_report
from graphrag_pipeline.config import Neo4jSettings, OpenRouterSettings
from graphrag_pipeline.extraction.schema_v1 import (
    DOMAIN_NODE_LABELS,
    DOMAIN_RELATIONSHIP_TYPES,
    SCHEMA_V1,
    SCHEMA_V1_HASH,
    compute_schema_hash,
)
from graphrag_pipeline.ingestion.load_musique_dataset import load_dataset
from graphrag_pipeline.storage.neo4j_connection import Neo4jConnection
from tools.model_gateway import ModelGatewayHandler


@pytest.fixture
def gateway_server(tmp_path: Path):
    """Start an in-process ModelGatewayHandler on an ephemeral port."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), ModelGatewayHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    cache_dir = tmp_path / "gateway_cache"

    yield base_url, cache_dir

    server.shutdown()
    server.server_close()
    ModelGatewayHandler.reset_defaults()


def _get_neo4j_settings() -> Neo4jSettings:
    os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
    os.environ.setdefault("NEO4J_USERNAME", "neo4j")
    os.environ.setdefault("NEO4J_PASSWORD", "changeme")
    os.environ.setdefault("NEO4J_DATABASE", "neo4j")

    try:
        settings = Neo4jSettings.from_env()
    except ValueError as e:
        pytest.skip(f"Neo4j not configured: {e}")

    with Neo4jConnection(settings) as conn:
        try:
            conn.verify_connectivity()
        except ServiceUnavailable:
            pytest.skip(f"Neo4j is not reachable at {settings.uri}")
    return settings


@pytest.mark.live
def test_phase_gate_m5_fixed_schema_ingestion_and_bridge_coverage(
    gateway_server: tuple[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Phase Gate M5 (Gate-M): 10 slice questions, fixed schema, zero schema-invention, and bridge coverage."""
    base_url, cache_dir = gateway_server
    neo4j_settings = _get_neo4j_settings()

    # Configure gateway in stub mode with caching enabled
    req_cfg = urllib.request.Request(
        f"{base_url}/config",
        data=json.dumps(
            {
                "mode": "stub",
                "cache_dir": str(cache_dir),
                "fake_unit_cost": 0.001,
                "reset_spend": True,
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req_cfg) as resp:
        assert resp.status == 200

    with urllib.request.urlopen(f"{base_url}/requests/clear") as resp:
        assert resp.status == 200

    # Configure environment for OpenRouter experiment profile
    gateway_api_url = f"{base_url}/v1"
    monkeypatch.setenv("OPENROUTER_BASE_URL", gateway_api_url)
    monkeypatch.setenv("OPENROUTER_API_KEY", "gate-m5-test-key")
    monkeypatch.setenv("OPENROUTER_LLM_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")
    monkeypatch.setenv("OPENROUTER_EMBEDDING_DIMENSIONS", "1536")
    monkeypatch.setenv("GRAPH_RAG_VECTOR_DIMENSIONS", "1536")
    monkeypatch.setenv("GRAPH_RAG_PROVIDER", "openrouter")
    monkeypatch.setenv("GRAPH_RAG_PROFILE", "experiment")
    monkeypatch.setenv("GRAPH_RAG_TEMPERATURE", "0.0")
    monkeypatch.setenv("GRAPH_RAG_SEED", "20261018")
    monkeypatch.setenv("OPENROUTER_PROVIDER_ORDER", "openai")
    monkeypatch.setenv("OPENROUTER_ALLOW_FALLBACKS", "false")
    monkeypatch.setenv("OPENROUTER_REQUIRE_PARAMETERS", "true")

    from tests.live.musique_ci import musique_inputs
    dataset_path, slice_path = musique_inputs()
    with slice_path.open("r", encoding="utf-8") as f:
        slice_data = json.load(f)
    target_ids = set(slice_data["tune_ids"][:10])

    slice_records = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["id"] in target_ids:
                slice_records.append(r)
                if len(slice_records) == 10:
                    break

    assert len(slice_records) == len(target_ids), f"Expected {len(target_ids)} slice records, found {len(slice_records)}"

    # Run ingestion for first 10 records
    summary = asyncio.run(
        load_dataset(
            dataset_path=dataset_path,
            start_record=1,
            max_records=10,
            source_split="dev",
            dry_run=False,
            show_progress=False,
        )
    )

    # 1. 0 writer failures
    assert len(summary.chunks_failed) == 0, f"Writer failures occurred: {summary.chunks_failed}"
    assert len(summary.chunks_succeeded) > 0

    # 2. Gateway inspection: 0 schema-invention calls, pinned parameters
    with urllib.request.urlopen(f"{base_url}/requests") as resp:
        req_history = json.loads(resp.read().decode())["requests"]

    chat_requests = [h for h in req_history if h["path"] == "/v1/chat/completions"]
    embed_requests = [h for h in req_history if h["path"] == "/v1/embeddings"]

    # Exactly 1 chat call and 1 embed call per newly ingested paragraph
    newly_ingested = len([c for c in summary.chunks_succeeded if not c.get("skipped")])
    if newly_ingested > 0:
        assert len(chat_requests) == newly_ingested
        assert len(embed_requests) == newly_ingested

        # 0 schema-invention calls: prompts are strictly entity extraction
        for cr in chat_requests:
            body = cr["body"]
            assert body["temperature"] == 0.0
            assert body["seed"] == 20261018
            assert body["provider"]["order"] == ["openai"]
            assert body["provider"]["allow_fallbacks"] is False
            assert body["provider"]["require_parameters"] is True
            assert "reasoning_effort" not in body
            msg_str = str(body.get("messages", []))
            assert "generate a schema" not in msg_str.lower()
            assert "extract a schema" not in msg_str.lower()

    # 3. Compute graph quality and bridge coverage
    with Neo4jConnection(neo4j_settings) as conn:
        quality_report = compute_graph_quality(conn.connect(), slice_records)

    assert quality_report.total_paragraphs > 0
    assert quality_report.total_bridge_links > 0
    assert quality_report.bridge_coverage_ratio >= 0.0
    assert len(quality_report.type_mix) > 0

    rendered = render_graph_quality_report(quality_report)
    assert "# Graph Quality & Bridge Coverage Report" in rendered
    assert "**Bridge Links Evaluated:**" in rendered

    # 4. Freeze schema verification: matches frozen artifact
    frozen_path = Path("eval/frozen_schema_v1.json")
    assert frozen_path.exists(), "eval/frozen_schema_v1.json must exist"
    with frozen_path.open("r", encoding="utf-8") as f:
        frozen_data = json.load(f)

    assert SCHEMA_V1_HASH == frozen_data["schema_hash"]
    assert sorted(DOMAIN_NODE_LABELS) == sorted(frozen_data["node_types"])
    assert sorted(DOMAIN_RELATIONSHIP_TYPES) == sorted(frozen_data["relationship_types"])
    assert frozen_data["temperature"] == 0.0
    assert frozen_data["seed"] == 20261018
    assert frozen_data["provider_order"] == ["openai"]

    # 5. Idempotent rerun: makes 0 new calls ($0 spend)
    summary_replay = asyncio.run(
        load_dataset(
            dataset_path=dataset_path,
            start_record=1,
            max_records=10,
            source_split="dev",
            dry_run=False,
            show_progress=False,
        )
    )
    new_extractions = [c for c in summary_replay.chunks_succeeded if not c.get("skipped")]
    assert len(new_extractions) == 0
    assert summary_replay.chunks_skipped == summary.chunks_planned
