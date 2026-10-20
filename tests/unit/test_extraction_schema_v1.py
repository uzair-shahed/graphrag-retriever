"""Unit tests for curated extraction schema v1 and pipeline integration (Step M5.1)."""

from __future__ import annotations

import asyncio
import json
import threading
import urllib.request
import unittest.mock as mock
from http.server import ThreadingHTTPServer
from pathlib import Path

import neo4j
import pytest
from neo4j_graphrag.components.schema import GraphSchema, NodeType, RelationshipType

from graphrag_pipeline.config import Neo4jSettings, OllamaSettings, OpenRouterSettings
from graphrag_pipeline.extraction.pipeline import build_simple_kg_pipeline
from graphrag_pipeline.extraction.schema_v1 import (
    DOMAIN_NODE_LABELS,
    DOMAIN_RELATIONSHIP_TYPES,
    RESERVED_NODE_LABELS,
    RESERVED_RELATIONSHIP_TYPES,
    SCHEMA_V1,
    SCHEMA_V1_HASH,
    build_schema_v1,
    compute_schema_hash,
    validate_schema_no_reserved_names,
)
from graphrag_pipeline.storage.neo4j_connection import Neo4jConnection
from tools.model_gateway import ModelGatewayHandler


def test_schema_v1_domain_types_and_strict_pruning() -> None:
    """Verify SCHEMA_V1 contains only curated domain types with strict pruning enabled."""
    schema = build_schema_v1()
    node_labels = {nt.label for nt in schema.node_types}
    assert node_labels == set(DOMAIN_NODE_LABELS)

    # Every node type has a 'name' property
    for nt in schema.node_types:
        prop_names = [p.name for p in (nt.properties or [])]
        assert "name" in prop_names

    rel_labels = {rt.label for rt in schema.relationship_types}
    assert rel_labels == set(DOMAIN_RELATIONSHIP_TYPES)

    # Strict pruning must be active
    assert schema.additional_node_types is False
    assert schema.additional_relationship_types is False


def test_schema_v1_deterministic_hash() -> None:
    """Verify compute_schema_hash is deterministic and invariant to internal ordering."""
    h1 = compute_schema_hash(SCHEMA_V1)
    h2 = compute_schema_hash(build_schema_v1())
    assert h1 == h2
    assert h1 == SCHEMA_V1_HASH
    assert h1.startswith("schema_v1_")


def test_schema_rejects_reserved_lexical_names() -> None:
    """The schema validator must reject Document, Chunk, and lexical edge names."""
    # SCHEMA_V1 itself must pass
    validate_schema_no_reserved_names(SCHEMA_V1)

    # Reserved node labels: Document, Chunk
    for reserved in ["Document", "document", "Chunk", "chunk"]:
        bad_schema = GraphSchema(
            node_types=(NodeType(label=reserved),),
            additional_node_types=False,
            additional_relationship_types=False,
        )
        with pytest.raises(ValueError, match=f"reserved node label '{reserved}'"):
            validate_schema_no_reserved_names(bad_schema)

    # Reserved relationship types: FROM_CHUNK, FROM_DOCUMENT, NEXT_CHUNK
    for reserved_rel in ["FROM_CHUNK", "from_chunk", "FROM_DOCUMENT", "NEXT_CHUNK"]:
        bad_rel_schema = GraphSchema(
            node_types=(NodeType(label="Person"),),
            relationship_types=(RelationshipType(label=reserved_rel),),
            additional_node_types=False,
            additional_relationship_types=False,
        )
        with pytest.raises(ValueError, match=f"reserved relationship type '{reserved_rel}'"):
            validate_schema_no_reserved_names(bad_rel_schema)


def test_stub_pipeline_makes_exactly_two_calls_and_zero_schema_invention() -> None:
    """With fixed schema, ingesting 1 paragraph makes 0 schema-invention calls and 2 model calls."""
    try:
        neo4j_settings = Neo4jSettings.from_env()
    except ValueError:
        pytest.skip("Local Neo4j instance not configured.")

    server = ThreadingHTTPServer(("127.0.0.1", 0), ModelGatewayHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"

    try:
        settings = OpenRouterSettings(
            api_key="test-key",
            base_url=f"{base_url}/v1",
            llm_model="openai/gpt-4o-mini",
            embedding_model="openai/text-embedding-3-small",
            embedding_dimensions=1536,
            llm_max_tokens=256,
            require_parameters=True,
            allow_fallbacks=False,
            profile="experiment",
            temperature=0.0,
            seed=20261018,
            provider_order=["openai"],
        )

        with Neo4jConnection(neo4j_settings) as conn:
            pipe = build_simple_kg_pipeline(conn.connect(), settings)

            with urllib.request.urlopen(f"{base_url}/requests/clear") as resp:
                assert resp.status == 200

            text = "Aster Systems acquired Omega Corp in 2021."
            asyncio.run(pipe.run_async(text=text))

            with urllib.request.urlopen(f"{base_url}/requests") as resp:
                history = json.loads(resp.read().decode())["requests"]

            # Exactly 2 calls: 1 embedding, 1 extraction
            assert len(history) == 2, f"Expected 2 calls per paragraph, got {len(history)}"

            paths = [h["path"] for h in history]
            assert "/v1/embeddings" in paths
            assert "/v1/chat/completions" in paths

            # 0 schema-invention requests: prompt must NOT be schema extraction
            chat_requests = [h for h in history if h["path"] == "/v1/chat/completions"]
            assert len(chat_requests) == 1
            chat_body = chat_requests[0]["body"]
            prompt_content = str(chat_body.get("messages", []))
            assert "Extract entities" in prompt_content or "Input text:" in prompt_content
            # SchemaFromTextExtractor prompt includes "generate a schema" or "extract a schema"
            assert "generate a schema" not in prompt_content.lower()
            assert "extract a schema" not in prompt_content.lower()
    finally:
        server.shutdown()
        server.server_close()
        ModelGatewayHandler.reset_defaults()


def test_openrouter_and_ollama_builders_receive_same_schema_hash() -> None:
    """Verify that both OpenRouter and Ollama pipeline builders configure the identical schema hash."""
    driver = mock.create_autospec(neo4j.Driver, instance=True)
    driver.execute_query.return_value = (
        [{"versions": ["5.26.0"], "edition": "community"}],
        mock.MagicMock(),
        [],
    )

    openrouter_settings = OpenRouterSettings(
        api_key="fake-key",
        base_url="http://127.0.0.1:9999/v1",
        llm_model="openai/gpt-4o-mini",
        embedding_model="openai/text-embedding-3-small",
        embedding_dimensions=1536,
        llm_max_tokens=256,
        require_parameters=True,
        allow_fallbacks=False,
    )
    ollama_settings = OllamaSettings(
        host="http://127.0.0.1:11434",
        llm_model="llama3.1:8b",
        embedding_model="nomic-embed-text",
    )

    pipe_or = build_simple_kg_pipeline(driver, openrouter_settings)
    pipe_ol = build_simple_kg_pipeline(driver, ollama_settings)

    schema_or = pipe_or.runner.config.schema_
    schema_ol = pipe_ol.runner.config.schema_

    assert schema_or is not None
    assert schema_ol is not None
    hash_or = compute_schema_hash(schema_or)
    hash_ol = compute_schema_hash(schema_ol)

    assert hash_or == hash_ol == SCHEMA_V1_HASH

