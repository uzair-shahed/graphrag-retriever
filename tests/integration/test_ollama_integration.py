"""Integration smoke tests for the configured local Ollama models.

Last modified: 18 October 2026

Purpose: Verify that the configured Ollama service can generate embeddings and
responses matching the project's answer and Neo4j graph schemas. These tests
do not write to Neo4j and are skipped when Ollama or configured models are
unavailable.

Caveats: Schema-constrained output validates structure, not graph quality or
answer grounding.
"""

from __future__ import annotations

import asyncio

import pytest
from ollama import Client
from neo4j_graphrag.components.types import Neo4jGraph
from neo4j_graphrag.types import LLMMessage

from graphrag_pipeline.answer import GeneratedAnswer
from graphrag_pipeline.config import OllamaSettings
from graphrag_pipeline.ollama_json_schema import OllamaJSONSchemaLLM
from neo4j_graphrag.embeddings.ollama import OllamaEmbeddings


def _load_ollama_or_skip() -> tuple[OllamaSettings, Client]:
    """Load Ollama settings and verify the configured service and models.

    Returns:
        The configured settings and a connected Ollama client.

    Raises:
        pytest.skip.Exception: If settings, the Ollama service, or a configured
            model is unavailable.
    """
    try:
        settings = OllamaSettings.from_env()
    except ValueError as error:
        pytest.skip(f"Ollama settings are not configured: {error}")

    client = Client(host=settings.host)
    try:
        models = {model.model for model in client.list().models}
    except Exception as error:  # noqa: BLE001 - integration availability check
        pytest.skip(f"Ollama is not reachable at {settings.host}: {error}")

    configured_models = {settings.llm_model, settings.embedding_model}
    missing_models = {
        model
        for model in configured_models
        if model not in models and f"{model}:latest" not in models
    }
    if missing_models:
        pytest.skip(f"Ollama models are not installed: {sorted(missing_models)}")
    return settings, client


def test_ollama_embedding_wrapper_generates_vector() -> None:
    """Verify the configured GraphRAG Ollama embedder returns a vector."""
    settings, _ = _load_ollama_or_skip()
    embedder = OllamaEmbeddings(model=settings.embedding_model, host=settings.host)

    vector = embedder.embed_query("Aster Systems acquired Northwind Labs in 2021.")

    assert vector
    assert all(isinstance(value, float) for value in vector)
    assert len(vector) > 1


def test_ollama_llm_generates_schema_constrained_answer() -> None:
    """Verify Ollama returns an answer matching the application's JSON schema."""
    settings, _ = _load_ollama_or_skip()
    llm = OllamaJSONSchemaLLM(
        model_name=settings.llm_model,
        response_model=GeneratedAnswer,
        model_params={"options": {"temperature": 0}},
        host=settings.host,
    )

    response = llm.invoke("Return answer 'ok' and cite chunk-1. Use exactly those values.")
    generated = GeneratedAnswer.model_validate_json(response.content)

    assert generated.answer.strip()
    assert generated.citations == ["chunk-1"]


def test_ollama_llm_generates_schema_constrained_neo4j_graph() -> None:
    """Verify async extraction output validates against Neo4jGraph."""
    settings, _ = _load_ollama_or_skip()
    llm = OllamaJSONSchemaLLM(
        model_name=settings.llm_model,
        response_model=Neo4jGraph,
        model_params={"options": {"temperature": 0}},
        host=settings.host,
    )

    response = asyncio.run(
        llm.ainvoke(
            [
                LLMMessage(
                    role="user",
                    content="Extract a graph from this text: Alice founded Acme Corporation.",
                )
            ],
            response_format=Neo4jGraph,
        )
    )
    graph = Neo4jGraph.model_validate_json(response.content)

    assert graph.nodes
    assert graph.relationships
