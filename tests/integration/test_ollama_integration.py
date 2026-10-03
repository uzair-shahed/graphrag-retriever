"""Integration smoke tests for the configured local Ollama models.

Last modified: 2 October 2026

Purpose: Verify that the GraphRAG Ollama wrappers can reach the configured
service, generate an embedding, and generate extraction-compatible text.
These tests are used by the Ollama integration test suite and do not write to
Neo4j. They are skipped when Ollama or the configured models are unavailable.

Caveats: The LLM test checks connectivity and a non-empty response, not graph
quality. OllamaLLM structured output is not tested because the installed
GraphRAG Ollama wrapper does not support it.
"""

from __future__ import annotations

import pytest
from ollama import Client
from ollama import ResponseError

from graphrag_pipeline.config import OllamaSettings
from neo4j_graphrag.embeddings.ollama import OllamaEmbeddings
from neo4j_graphrag.llm import OllamaLLM


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


def test_ollama_llm_wrapper_generates_text() -> None:
    """Verify the configured GraphRAG Ollama LLM returns non-empty text."""
    settings, _ = _load_ollama_or_skip()
    llm = OllamaLLM(
        model_name=settings.llm_model,
        model_params={"options": {"temperature": 0}},
        host=settings.host,
    )

    response = llm.invoke(
        "Return only this JSON object: {\"status\": \"ok\"}."
    )

    assert response.content.strip()
