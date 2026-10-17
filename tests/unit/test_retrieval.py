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
        query_text="Which company acquired Northwind Labs?", top_k=12
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


def test_retrieve_uses_graphrag_search_and_retains_generated_answer() -> None:
    """GraphRAG search returns bounded context and avoids a second answer call."""
    graphrag = MagicMock()
    graphrag.search.return_value = SimpleNamespace(
        answer="Aster Systems",
        retriever_result=SimpleNamespace(items=[evidence_item()]),
    )

    result = retrieve(
        request(),
        graphrag,
        retrieval_settings=RetrievalSettings(backend="graphrag", mode="vector_cypher", top_k=3),
    )

    graphrag.search.assert_called_once_with(
        query_text="Which company acquired Northwind Labs?",
        retriever_config={
            "top_k": 3,
            "query_params": {"node_limit": 200, "query_limit": 201},
        },
        return_context=True,
    )
    assert result.backend == "graphrag"
    assert result.mode == "graphrag"
    assert result.generated_answer == "Aster Systems"
    assert result.evidence[0].chunk_id == "chunk-1"


def test_retrieve_deduplicates_graph_rows_before_context_limit() -> None:
    """Duplicate Neo4j rows do not consume the bounded evidence context."""
    retriever = MagicMock()
    retriever.search.return_value = SimpleNamespace(
        items=[evidence_item("chunk-1"), evidence_item("chunk-1"), evidence_item("chunk-2")]
    )

    result = retrieve(
        request(),
        retriever,
        retrieval_settings=RetrievalSettings(top_k=2, max_context_chunks=2),
    )

    assert [item.chunk_id for item in result.evidence] == ["chunk-1", "chunk-2"]


def test_retrieve_disambiguates_musique_query_with_record_titles() -> None:
    """MuSiQue retrieval uses source titles without exposing answer labels."""
    musique_request = request().model_copy(
        update={
            "origin": "musique",
            "record_id": "record-1",
            "paragraph_titles": ["Green", "Green (Steve Hillage album)", "Miquette Giraudy"],
        }
    )
    retriever = MagicMock()
    retriever.search.return_value = SimpleNamespace(items=[evidence_item()])

    retrieve(musique_request, retriever)

    call = retriever.search.call_args.kwargs
    assert "Which company acquired Northwind Labs?" in call["query_text"]
    assert "Relevant source titles: Green (Steve Hillage album)" in call["query_text"]
    assert call["filters"] == {"record_id": "record-1"}


def test_retrieve_keeps_empty_graphrag_context_as_insufficient_evidence() -> None:
    """GraphRAG cannot produce a successful answer without returned context."""
    graphrag = MagicMock()
    graphrag.search.return_value = SimpleNamespace(
        answer="unsupported",
        retriever_result=SimpleNamespace(items=[]),
    )

    result = retrieve(
        request(),
        graphrag,
        retrieval_settings=RetrievalSettings(backend="graphrag", mode="vector_cypher"),
    )

    assert result.evidence == []
    assert result.generated_answer is None
    assert result.backend == "graphrag"


def test_build_vector_retriever_uses_openrouter_embedder() -> None:
    """Vector mode uses VectorCypherRetriever with the Document-hop-only projection."""
    with (
        patch("graphrag_pipeline.retrieval.OpenAIEmbeddings") as embedder,
        patch("graphrag_pipeline.retrieval.VectorCypherRetriever") as retriever,
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
    query = retriever.call_args.kwargs["retrieval_query"]
    assert "FROM_DOCUMENT" in query
    assert "doc.chunk_id" in query
    assert "apoc.path.subgraphAll" not in query


def test_build_vector_cypher_retriever_uses_ollama_embedder() -> None:
    """Vector-Cypher mode uses the custom projection retriever and an explicit relationship allowlist."""
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
            relationship_types=["DEVELOPED_BY", "FROM_DOCUMENT", "FROM_CHUNK", "bad type"],
        )

    embedder.assert_called_once_with(model="test-embedding", host="http://localhost:11434")
    query = retriever.call_args.kwargs["retrieval_query"]
    assert "RETURN node.text" in query
    assert "FROM_DOCUMENT" in query
    assert "apoc.path.subgraphAll" in query
    assert "FROM_CHUNK<|DEVELOPED_BY" in query
    assert "bad type" not in query


def test_build_vector_cypher_retriever_discovers_relationship_types_from_driver() -> None:
    """Without an override, the retriever discovers relationship types from the live graph."""
    settings = OllamaSettings(
        host="http://localhost:11434",
        llm_model="test-llm",
        embedding_model="test-embedding",
    )
    driver = MagicMock()
    session = driver.session.return_value.__enter__.return_value
    session.run.return_value.data.return_value = [
        {"relationshipType": "DEVELOPED_BY"},
        {"relationshipType": "FROM_DOCUMENT"},
        {"relationshipType": "FROM_CHUNK"},
    ]
    with (
        patch("graphrag_pipeline.retrieval.OllamaEmbeddings"),
        patch("graphrag_pipeline.retrieval.VectorCypherRetriever") as retriever,
    ):
        build_retriever(
            driver,
            "neo4j",
            settings,
            retrieval_settings=RetrievalSettings(mode="vector_cypher"),
        )

    driver.session.assert_called_once_with(database="neo4j")
    query = retriever.call_args.kwargs["retrieval_query"]
    assert "FROM_CHUNK<|DEVELOPED_BY" in query


def test_retrieval_settings_rejects_non_positive_max_graph_nodes() -> None:
    """max_graph_nodes is a resource budget and must be positive."""
    with pytest.raises(ValueError, match="max_graph_nodes must be positive"):
        RetrievalSettings(max_graph_nodes=0)


def test_retrieval_settings_rejects_graphrag_backend_without_vector_cypher_mode() -> None:
    """GraphRAG's full-component traversal requires the vector_cypher Cypher projection."""
    with pytest.raises(ValueError, match="graphrag backend requires vector_cypher"):
        RetrievalSettings(backend="graphrag", mode="vector")


def graph_evidence_item(
    graph_nodes=None,
    graph_relationships=None,
    graph_complete=None,
):
    """Build one fake retriever item carrying entity-subgraph metadata."""
    metadata = {
        "chunk_id": "chunk-1",
        "document_id": "document-1",
        "source_file": "fixture.jsonl",
        "source_line": "2",
        "score": 0.9,
    }
    if graph_nodes is not None:
        metadata["graph_nodes"] = graph_nodes
    if graph_relationships is not None:
        metadata["graph_relationships"] = graph_relationships
    if graph_complete is not None:
        metadata["graph_complete"] = graph_complete
    return SimpleNamespace(content="Evidence text", metadata=metadata)


def test_retrieve_maps_graph_nodes_and_relationships() -> None:
    """A seed chunk's entity subgraph is parsed into validated graph evidence."""
    retriever = MagicMock()
    retriever.search.return_value = SimpleNamespace(
        items=[
            graph_evidence_item(
                graph_nodes=[{"id": "n1", "labels": ["Person"], "name": "Alice"}],
                graph_relationships=[{"type": "WORKS_AT", "start_node_id": "n1", "end_node_id": "n2"}],
                graph_complete=True,
            )
        ]
    )

    result = retrieve(request(), retriever, retrieval_settings=RetrievalSettings(mode="vector_cypher"))

    evidence = result.evidence[0]
    assert evidence.graph_nodes[0].node_id == "n1"
    assert evidence.graph_nodes[0].name == "Alice"
    assert evidence.graph_relationships[0].type == "WORKS_AT"
    assert evidence.graph_component_complete is True


def test_retrieve_defaults_graph_fields_for_vector_mode() -> None:
    """VectorRAG's no-expansion projection never carries graph metadata."""
    retriever = MagicMock()
    retriever.search.return_value = SimpleNamespace(items=[graph_evidence_item()])

    result = retrieve(request(), retriever, retrieval_settings=RetrievalSettings(mode="vector"))

    evidence = result.evidence[0]
    assert evidence.graph_nodes == []
    assert evidence.graph_relationships == []
    assert evidence.graph_component_complete is True


def test_retrieve_rejects_malformed_graph_node() -> None:
    """A graph node missing its identity is an explicit retrieval failure."""
    retriever = MagicMock()
    retriever.search.return_value = SimpleNamespace(
        items=[graph_evidence_item(graph_nodes=[{"labels": ["Person"]}])]
    )

    with pytest.raises(RetrievalError, match="invalid graph node"):
        retrieve(request(), retriever, retrieval_settings=RetrievalSettings(mode="vector_cypher"))


def test_retrieve_rejects_malformed_graph_relationship() -> None:
    """A relationship missing an endpoint is an explicit retrieval failure."""
    retriever = MagicMock()
    retriever.search.return_value = SimpleNamespace(
        items=[graph_evidence_item(graph_relationships=[{"type": "WORKS_AT", "start_node_id": "n1"}])]
    )

    with pytest.raises(RetrievalError, match="invalid graph relationship"):
        retrieve(request(), retriever, retrieval_settings=RetrievalSettings(mode="vector_cypher"))


def test_retrieve_rejects_non_bool_graph_complete() -> None:
    """A malformed completeness flag must never be silently coerced."""
    retriever = MagicMock()
    retriever.search.return_value = SimpleNamespace(items=[graph_evidence_item(graph_complete="yes")])

    with pytest.raises(RetrievalError, match="invalid graph_complete"):
        retrieve(request(), retriever, retrieval_settings=RetrievalSettings(mode="vector_cypher"))


def test_graphrag_seed_count_never_exceeds_the_evidence_bound() -> None:
    """GraphRAG is not over-fetched, so its answer cannot see candidates evidence omits."""
    graphrag = MagicMock()
    graphrag.search.return_value = SimpleNamespace(
        answer="Aster Systems",
        retriever_result=SimpleNamespace(items=[evidence_item("chunk-1"), evidence_item("chunk-2")]),
    )
    settings = RetrievalSettings(
        backend="graphrag", mode="vector_cypher", top_k=10, max_context_chunks=2
    )

    result = retrieve(request(), graphrag, retrieval_settings=settings)

    assert graphrag.search.call_args.kwargs["retriever_config"]["top_k"] == 2
    assert [item.chunk_id for item in result.evidence] == ["chunk-1", "chunk-2"]


def test_graphrag_rejects_answer_context_larger_than_retained_evidence() -> None:
    """An over-fetched candidate holding the answer is an explicit failure, never hidden support."""
    graphrag = MagicMock()
    graphrag.search.return_value = SimpleNamespace(
        answer="Answer found only in chunk-3",
        retriever_result=SimpleNamespace(
            items=[evidence_item("chunk-1"), evidence_item("chunk-2"), evidence_item("chunk-3")]
        ),
    )
    settings = RetrievalSettings(
        backend="graphrag", mode="vector_cypher", top_k=2, max_context_chunks=2
    )

    with pytest.raises(RetrievalError, match="exceeding max_context_chunks=2"):
        retrieve(request(), graphrag, retrieval_settings=settings)


def test_graphrag_duplicate_context_rows_do_not_hide_evidence() -> None:
    """Duplicate rows collapse to one evidence chunk without exceeding the bound."""
    graphrag = MagicMock()
    graphrag.search.return_value = SimpleNamespace(
        answer="Aster Systems",
        retriever_result=SimpleNamespace(items=[evidence_item("chunk-1"), evidence_item("chunk-1")]),
    )
    settings = RetrievalSettings(
        backend="graphrag", mode="vector_cypher", top_k=2, max_context_chunks=2
    )

    result = retrieve(request(), graphrag, retrieval_settings=settings)

    assert [item.chunk_id for item in result.evidence] == ["chunk-1"]


def formatted_record(chunk_id: str, text: str, **extra) -> dict:
    """Build one raw Neo4j-style row for the retriever result formatter."""
    return {"chunk_id": chunk_id, "text": text, "document_id": "document-1", "score": 0.5, **extra}


def test_format_record_renders_model_context_and_keeps_raw_text_in_metadata() -> None:
    """Managed GraphRAG context shows citable chunk IDs while evidence text stays raw."""
    from graphrag_pipeline.retrieval import _format_record

    item = _format_record(formatted_record("chunk-1", "Raw paragraph text."))

    assert item.content == "[chunk-1] Raw paragraph text.\n"
    assert item.metadata["text"] == "Raw paragraph text."
    evidence = retrieve_evidence_from_items([item])
    assert evidence[0].text == "Raw paragraph text."


def retrieve_evidence_from_items(items):
    """Run the retrieval adapter over fake items and return its evidence."""
    retriever = MagicMock()
    retriever.search.return_value = SimpleNamespace(items=items)
    return retrieve(request(), retriever).evidence


def test_format_record_context_matches_vectorrag_prompt_evidence_rendering() -> None:
    """GraphRAG.search joins row content with newlines; that equals the VectorRAG context."""
    from graphrag_pipeline.answer_context import render_evidence_context
    from graphrag_pipeline.retrieval import _format_record

    nodes = [{"id": "n1", "labels": ["Person"], "name": "Alice"}, {"id": "n2", "labels": ["Company"], "name": "Acme"}]
    rels = [{"type": "WORKS_AT", "start_node_id": "n1", "end_node_id": "n2"}]
    rows = [
        formatted_record("chunk-1", "Alice works at Acme.", graph_nodes=nodes, graph_relationships=rels, graph_complete=True),
        formatted_record("chunk-2", "Second paragraph."),
    ]

    items = [_format_record(row) for row in rows]
    managed_context = "\n".join(item.content for item in items)
    evidence = retrieve_evidence_from_items(items)

    assert managed_context == render_evidence_context(evidence)
    assert "Entities in this chunk: Alice (Person); Acme (Company)" in managed_context
    assert "Relationships in this chunk: Alice -[WORKS_AT]-> Acme" in managed_context


def test_format_record_marks_truncated_graph_component() -> None:
    """A size-limited component is labelled so the model is not told it is complete."""
    from graphrag_pipeline.retrieval import _format_record

    row = formatted_record(
        "chunk-1",
        "Text.",
        graph_nodes=[{"id": "n1", "labels": ["Person"], "name": "Alice"}],
        graph_relationships=[],
        graph_complete=False,
    )

    assert "truncated by a size limit" in _format_record(row).content


def test_build_graphrag_uses_the_shared_answer_template() -> None:
    """The managed GraphRAG prompt uses the same instructions as the VectorRAG prompt."""
    from graphrag_pipeline.answer_context import ANSWER_PROMPT_TEMPLATE, ANSWER_SYSTEM_INSTRUCTIONS
    from graphrag_pipeline.retrieval import build_graphrag

    with patch("graphrag_pipeline.retrieval.GraphRAG") as graphrag_cls:
        build_graphrag("retriever", "llm")

    template = graphrag_cls.call_args.kwargs["prompt_template"]
    assert template.template == ANSWER_PROMPT_TEMPLATE
    assert template.system_instructions == ANSWER_SYSTEM_INSTRUCTIONS
    assert "context" in template.expected_inputs and "query_text" in template.expected_inputs


def test_graphrag_blank_answer_is_kept_for_contract_validation() -> None:
    """A blank managed answer reaches the answer boundary instead of a retrieval error."""
    graphrag = MagicMock()
    graphrag.search.return_value = SimpleNamespace(
        answer="",
        retriever_result=SimpleNamespace(items=[evidence_item()]),
    )
    settings = RetrievalSettings(backend="graphrag", mode="vector_cypher")

    result = retrieve(request(), graphrag, retrieval_settings=settings)

    assert result.generated_answer == ""
    assert [item.chunk_id for item in result.evidence] == ["chunk-1"]
