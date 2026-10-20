"""Unit tests for Ollama native JSON Schema request wiring."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

from neo4j_graphrag.components.types import Neo4jGraph
from neo4j_graphrag.types import LLMMessage

from graphrag_pipeline.ollama_json_schema import OllamaJSONSchemaLLM


def llm() -> OllamaJSONSchemaLLM:
    """Build a local wrapper without making a provider request."""
    return OllamaJSONSchemaLLM(
        model_name="test-model",
        response_model=Neo4jGraph,
        model_params={"think": False, "keep_alive": "10m", "options": {"temperature": 0, "num_predict": 512}},
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
    with patch.object(model.client, "_request", return_value=fake_response(content)) as chat:
        response = model.invoke("Extract a graph.", response_format={"type": "json_object"})

    payload = chat.call_args.kwargs["json"]
    assert chat.call_args.args == (model.ollama.ChatResponse, "POST", "/api/chat")
    assert payload["format"] == Neo4jGraph.model_json_schema()
    assert payload["options"] == {"temperature": 0, "num_predict": 512}
    assert payload["think"] is False
    assert payload["keep_alive"] == "10m"
    assert "format" not in payload["options"]
    assert Neo4jGraph.model_validate_json(response.content).nodes == []
    assert response.usage.total_tokens == 8


def test_async_structured_call_passes_requested_pydantic_schema() -> None:
    """GraphRAG's structured extraction call reaches Ollama's schema field."""
    model = llm()
    content = '{"nodes": [], "relationships": []}'
    with patch.object(
        model.async_client,
        "_request",
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
    payload = chat.await_args.kwargs["json"]
    assert payload["format"] == Neo4jGraph.model_json_schema()
    assert payload["options"] == {"temperature": 0, "num_predict": 512}
    assert payload["think"] is False
    assert payload["keep_alive"] == "10m"
    assert Neo4jGraph.model_validate_json(response.content).relationships == []


def test_locked_sdk_transport_sends_think_and_parses_response() -> None:
    """Exercise real SDK serialization and parsing with an offline HTTP transport."""
    model = llm()

    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert request.url.path == "/api/chat"
        assert payload["think"] is False
        assert payload["stream"] is False
        assert payload["messages"] == [{"role": "user", "content": "Extract a graph."}]
        assert payload["options"]["num_predict"] == 512
        return httpx.Response(200, json={
            "message": {"role": "assistant", "content": '{"nodes": [], "relationships": []}'},
            "done": True,
            "prompt_eval_count": 5,
            "eval_count": 3,
        })

    async def invoke_async() -> None:
        async with httpx.AsyncClient(
            base_url="http://localhost:11434", transport=httpx.MockTransport(respond)
        ) as client:
            with patch.object(model.async_client, "_client", client):
                response = await model.ainvoke("Extract a graph.")
        assert response.usage.total_tokens == 8

    with httpx.Client(
        base_url="http://localhost:11434", transport=httpx.MockTransport(respond)
    ) as client:
        with patch.object(model.client, "_client", client):
            response = model.invoke("Extract a graph.")
    assert response.usage.total_tokens == 8
    asyncio.run(invoke_async())
