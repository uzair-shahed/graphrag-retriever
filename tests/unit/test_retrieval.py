"""Deterministic tests for the Neo4j GraphRAG retrieval adapter."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from graphrag_pipeline.config import OllamaSettings, OpenRouterSettings
from graphrag_pipeline.query import QueryRequest
from graphrag_pipeline.retrieval import (
    RetrievalError,
    RetrievalSettings,
    build_retriever,
    retrieve,
)
from graphrag_pipeline.retrieval_prerequisites import VectorIndexSettings


def request() -> QueryRequest:
    """Return a deterministic validated query request."""
    return QueryRequest(
        question="Which company acquired Northwind Labs?",
        trace_id="trace-1",
        origin="custom",
    )


def openrouter_settings() -> OpenRouterSettings:
    """Return secret-free OpenRouter settings for constructor tests."""
    return OpenRouterSettings(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        llm_model="test-llm",
        embedding_model="test-embedding",
        embedding_dimensions=3,
        llm_max_tokens=100,
        require_parameters=True,
        allow_fallbacks=True,
    )


def evidence_item(chunk_id: str = "chunk-1", text: str = "Evidence text"):
    """Build one fake GraphRAG result item with provenance."""
    return SimpleNamespace(
        content=text,
        metadata={
            "chunk_id": chunk_id,
            "document_id": "document-1",
            "source_file": "fixture.jsonl",
            "source_line": "2",
            "score": 0.9,
            "graph_references": ["ACQUIRED"],
        },
    )


def test_retrieve_maps_structured_evidence_and_preserves_trace() -> None:
    """Retriever items become validated evidence with source metadata."""
    retriever = MagicMock()
    retriever.search.return_value = SimpleNamespace(items=[evidence_item()])

    result = retrieve(request(), retriever, retrieval_settings=RetrievalSettings(top_k=3))

    retriever.search.assert_called_once_with(
        query_text="Which company acquired Northwind Labs?", top_k=3
    )
    assert result.trace_id == "trace-1"
    assert result.question == request().question
    assert result.evidence[0].chunk_id == "chunk-1"
    assert result.evidence[0].source_line == 2
    assert result.evidence[0].graph_references == ["ACQUIRED"]


def test_retrieve_applies_context_limit() -> None:
    """The adapter bounds returned context independently of provider output."""
    retriever = MagicMock()
    retriever.search.return_value = SimpleNamespace(
        items=[evidence_item("chunk-1"), evidence_item("chunk-2")]
    )

    result = retrieve(
        request(),
        retriever,
        retrieval_settings=RetrievalSettings(top_k=5, max_context_chunks=1),
    )

    assert [item.chunk_id for item in result.evidence] == ["chunk-1"]


def test_retrieve_returns_empty_evidence_without_inventing_content() -> None:
    """An empty Neo4j result is represented as an empty evidence result."""
    retriever = MagicMock()
    retriever.search.return_value = SimpleNamespace(items=[])

    result = retrieve(request(), retriever)

    assert result.evidence == []
    assert not result.has_evidence


def test_retrieve_rejects_malformed_provider_item() -> None:
    """Missing chunk identity is an explicit malformed-response error."""
    retriever = MagicMock()
    retriever.search.return_value = SimpleNamespace(
        items=[SimpleNamespace(content="Text", metadata={})]
    )

    with pytest.raises(RetrievalError, match="without chunk_id"):
        retrieve(request(), retriever)


def test_retrieve_normalizes_provider_failure() -> None:
    """Neo4j/provider exceptions are visible as retrieval errors."""
    retriever = MagicMock()
    retriever.search.side_effect = RuntimeError("index unavailable")

    with pytest.raises(RetrievalError, match="index unavailable"):
        retrieve(request(), retriever)


def test_build_vector_retriever_uses_openrouter_embedder() -> None:
    """Vector mode uses the supported VectorRetriever and configured embedder."""
    with (
        patch("graphrag_pipeline.retrieval.OpenAIEmbeddings") as embedder,
        patch("graphrag_pipeline.retrieval.VectorRetriever") as retriever,
    ):
        build_retriever(
            MagicMock(),
            "neo4j",
            openrouter_settings(),
            index_settings=VectorIndexSettings(name="index", dimensions=3),
            retrieval_settings=RetrievalSettings(mode="vector"),
        )

    embedder.assert_called_once_with(
        model="test-embedding",
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
    )
    assert retriever.call_args.kwargs["index_name"] == "index"
    assert retriever.call_args.kwargs["return_properties"]


def test_build_vector_cypher_retriever_uses_ollama_embedder() -> None:
    """Vector-Cypher mode uses the supported custom projection retriever."""
    settings = OllamaSettings(
        host="http://localhost:11434",
        llm_model="test-llm",
        embedding_model="test-embedding",
    )
    with (
        patch("graphrag_pipeline.retrieval.OllamaEmbeddings") as embedder,
        patch("graphrag_pipeline.retrieval.VectorCypherRetriever") as retriever,
    ):
        build_retriever(
            MagicMock(),
            "neo4j",
            settings,
            retrieval_settings=RetrievalSettings(mode="vector_cypher"),
        )

    embedder.assert_called_once_with(model="test-embedding", host="http://localhost:11434")
    assert "RETURN node.text" in retriever.call_args.kwargs["retrieval_query"]


def test_retrieval_settings_defaults_to_vector_cypher() -> None:
    """The default retrieval settings must use vector_cypher mode."""
    settings = RetrievalSettings()
    assert settings.mode == "vector_cypher"
    assert settings.top_k == 5
    assert settings.max_context_chunks == 5


def test_retrieval_query_traverses_from_document() -> None:
    """The Cypher retrieval query must traverse FROM_DOCUMENT to Document node."""
    from graphrag_pipeline.retrieval import _retrieval_query

    query = _retrieval_query()
    assert "MATCH (node)-[:FROM_DOCUMENT]->(d:Document)" in query
    assert "d.paragraph_hash AS paragraph_hash" in query
    assert "coalesce(d.title, node.title) AS title" in query
    assert "score" in query


def test_format_record_requires_paragraph_hash() -> None:
    """Records missing paragraph_hash must raise RetrievalError."""
    from graphrag_pipeline.retrieval import _format_record

    record = {"text": "Valid text but no hash"}
    with pytest.raises(RetrievalError, match="missing paragraph_hash"):
        _format_record(record)


def test_format_record_builds_chunk_id_from_hash() -> None:
    """Records with paragraph_hash map to musique:paragraph:{hash} chunk_id."""
    from graphrag_pipeline.retrieval import _format_record

    record = {
        "text": "Valid paragraph content",
        "paragraph_hash": "deadbeef1234",
        "title": "Document Title",
        "score": 0.85,
    }
    item = _format_record(record)
    assert item.content == "Valid paragraph content"
    assert item.metadata["chunk_id"] == "musique:paragraph:deadbeef1234"
    assert item.metadata["paragraph_hash"] == "deadbeef1234"
    assert item.metadata["title"] == "Document Title"
    assert item.metadata["score"] == 0.85

