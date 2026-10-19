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
@pytest.mark.xfail(strict=True, reason="chunk_id is stored on Document, not Chunk", raises=RetrievalError)
def test_readme_flow_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run fixture ingestion, create vector index, and verify retrieval failure."""
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
    retrieval_settings = RetrievalSettings(mode="vector")

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
        # On HEAD (9f8cc3d), this raises RetrievalError because chunk_id on Chunk is None
        retrieve(request, retriever, retrieval_settings=retrieval_settings)
