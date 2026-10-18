"""Unit tests for Ollama native JSON Schema request wiring."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from neo4j_graphrag.components.types import Neo4jGraph
from neo4j_graphrag.types import LLMMessage

from graphrag_pipeline.ollama_json_schema import OllamaJSONSchemaLLM


def llm() -> OllamaJSONSchemaLLM:
    """Build a local wrapper without making a provider request."""
    return OllamaJSONSchemaLLM(
        model_name="test-model",
        response_model=Neo4jGraph,
        model_params={"options": {"temperature": 0}},
        host="http://localhost:11434",
    )


def fake_response(content: str) -> SimpleNamespace:
    """Return an Ollama-shaped response with token counts."""
    return SimpleNamespace(
        message=SimpleNamespace(content=content),
        prompt_eval_count=5,
        eval_count=3,
    )


def test_sync_call_passes_schema_as_top_level_format() -> None:
    """Synchronous string calls send the configured schema, not JSON mode."""
    model = llm()
    content = '{"nodes": [], "relationships": []}'
    with patch.object(model.client, "chat", return_value=fake_response(content)) as chat:
        response = model.invoke("Extract a graph.", response_format={"type": "json_object"})

    assert chat.call_args.kwargs["format"] == Neo4jGraph.model_json_schema()
    assert chat.call_args.kwargs["options"] == {"temperature": 0}
    assert "format" not in chat.call_args.kwargs["options"]
    assert Neo4jGraph.model_validate_json(response.content).nodes == []
    assert response.usage.total_tokens == 8


def test_async_structured_call_passes_requested_pydantic_schema() -> None:
    """GraphRAG's structured extraction call reaches Ollama's schema field."""
    model = llm()
    content = '{"nodes": [], "relationships": []}'
    with patch.object(
        model.async_client,
        "chat",
        new_callable=AsyncMock,
        return_value=fake_response(content),
    ) as chat:
        response = asyncio.run(
            model.ainvoke(
                [LLMMessage(role="user", content="Extract a graph.")],
                response_format=Neo4jGraph,
            )
        )

    assert model.supports_structured_output is True
    assert chat.await_args.kwargs["format"] == Neo4jGraph.model_json_schema()
    assert chat.await_args.kwargs["options"] == {"temperature": 0}
    assert Neo4jGraph.model_validate_json(response.content).relationships == []