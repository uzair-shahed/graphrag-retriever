"""Neo4j GraphRAG retrieval adapter for structured graph evidence.

Last modified: 2026-10-06

Purpose of this file:
    Construct the supported Neo4j ``VectorRetriever`` or
    ``VectorCypherRetriever``, submit a validated query, and map returned graph
    records into the application's ``EvidenceChunk`` contract.

Files that use this file:
    The future query command and retrieval-test program will call
    ``retrieve``. Unit tests mock the Neo4j retriever and embedding boundaries.

Caveats:
    This module retrieves evidence only. ``GraphRAG.search()`` answer
    generation and read-only prerequisite verification remain separate steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from neo4j_graphrag.embeddings.ollama import OllamaEmbeddings
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorCypherRetriever, VectorRetriever
from neo4j_graphrag.types import RetrieverResultItem

from graphrag_pipeline.config import OllamaSettings, OpenRouterSettings
from graphrag_pipeline.query import EvidenceChunk, QueryRequest
from graphrag_pipeline.retrieval_prerequisites import VectorIndexSettings


class RetrievalError(RuntimeError):
    """Raised when Neo4j returns an unusable retrieval result or provider error."""


@dataclass(frozen=True)
class RetrievalSettings:
    """Bounded configuration for one GraphRAG retrieval call."""

    mode: Literal["vector", "vector_cypher"] = "vector_cypher"
    top_k: int = 5
    max_context_chunks: int = 5

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.max_context_chunks < 1:
            raise ValueError("max_context_chunks must be positive")


@dataclass(frozen=True)
class RetrievalResult:
    """Evidence returned for one query, retaining its question and trace."""

    trace_id: str
    question: str
    mode: str
    evidence: list[EvidenceChunk]

    @property
    def has_evidence(self) -> bool:
        """Return whether at least one graph-backed chunk was returned."""
        return bool(self.evidence)


def _build_embedder(
    model_settings: OpenRouterSettings | OllamaSettings,
) -> Any:
    """Build the same provider embedding integration used by extraction."""
    if isinstance(model_settings, OpenRouterSettings):
        return OpenAIEmbeddings(
            model=model_settings.embedding_model,
            api_key=model_settings.api_key,
            base_url=model_settings.base_url,
        )
    return OllamaEmbeddings(model=model_settings.embedding_model, host=model_settings.host)


def _record_value(record: Any, key: str) -> Any:
    """Read a field from a Neo4j record or dictionary result."""
    if isinstance(record, dict):
        return record.get(key)
    if hasattr(record, "get"):
        return record.get(key)
    try:
        return record[key]
    except (KeyError, TypeError, IndexError):
        return None


def _format_record(record: Any) -> RetrieverResultItem:
    """Map a raw Neo4j record to text plus provenance metadata."""
    node = _record_value(record, "node")
    if isinstance(node, dict):
        values = dict(node)
    else:
        values = {
            key: _record_value(record, key)
            for key in (
                "paragraph_hash",
                "chunk_id",
                "text",
                "document_id",
                "record_id",
                "source_file",
                "source_line",
                "source_sha256",
                "paragraph_idx",
                "title",
            )
        }
    paragraph_hash = values.get("paragraph_hash") or _record_value(record, "paragraph_hash")
    if not isinstance(paragraph_hash, str) or not paragraph_hash:
        raise RetrievalError("retrieval result is missing paragraph_hash")

    chunk_id = f"musique:paragraph:{paragraph_hash}"
    values["paragraph_hash"] = paragraph_hash
    values["chunk_id"] = chunk_id

    text = values.get("text") or _record_value(record, "text")
    if not isinstance(text, str) or not text:
        raise RetrievalError(f"retrieval result for {chunk_id!r} is missing text")

    metadata = dict(values)
    metadata.update(
        {
            "score": _record_value(record, "score"),
            "graph_references": _record_value(record, "graph_references") or [],
        }
    )
    return RetrieverResultItem(content=text, metadata=metadata)


def _retrieval_query() -> str:
    """Return the bounded Cypher projection used by VectorCypherRetriever."""
    return (
        "MATCH (node)-[:FROM_DOCUMENT]->(d:Document) "
        "RETURN node.text AS text, "
        "d.paragraph_hash AS paragraph_hash, "
        "coalesce(d.title, node.title) AS title, "
        "coalesce(d.document_id, node.document_id) AS document_id, "
        "coalesce(d.record_id, node.record_id) AS record_id, "
        "coalesce(d.source_file, node.source_file) AS source_file, "
        "coalesce(d.source_line, node.source_line) AS source_line, "
        "coalesce(d.source_sha256, node.source_sha256) AS source_sha256, "
        "coalesce(d.paragraph_idx, node.paragraph_idx) AS paragraph_idx, "
        "score"
    )


def build_retriever(
    driver: Any,
    neo4j_database: str,
    model_settings: OpenRouterSettings | OllamaSettings,
    *,
    index_settings: VectorIndexSettings | None = None,
    retrieval_settings: RetrievalSettings | None = None,
) -> Any:
    """Build a configured Neo4j GraphRAG retriever.

    Args:
        driver: Neo4j driver passed directly to the supported retriever.
        neo4j_database: Database containing the verified vector index.
        model_settings: Existing OpenRouter or Ollama embedding settings.
        index_settings: Expected vector-index contract.
        retrieval_settings: Retrieval mode and bounds.

    Returns:
        A configured ``VectorRetriever`` or ``VectorCypherRetriever``.
    """
    index = index_settings or VectorIndexSettings.from_env()
    retrieval = retrieval_settings or RetrievalSettings()
    embedder = _build_embedder(model_settings)
    common = {
        "driver": driver,
        "index_name": index.name,
        "embedder": embedder,
        "neo4j_database": neo4j_database,
        "result_formatter": _format_record,
    }
    if retrieval.mode == "vector":
        return VectorRetriever(
            **common,
            return_properties=[
                "chunk_id",
                "text",
                "document_id",
                "record_id",
                "source_file",
                "source_line",
                "source_sha256",
                "paragraph_idx",
                "title",
            ],
        )
    if retrieval.mode == "vector_cypher":
        return VectorCypherRetriever(
            **common,
            retrieval_query=_retrieval_query(),
        )
    raise ValueError(f"unsupported retrieval mode: {retrieval.mode}")


def _evidence_from_item(item: Any) -> EvidenceChunk:
    """Validate one GraphRAG retriever item as an application evidence chunk."""
    content = getattr(item, "content", None)
    metadata = getattr(item, "metadata", None) or {}
    if not isinstance(content, str) or not content:
        raise RetrievalError("retriever returned evidence without text")
    chunk_id = metadata.get("chunk_id")
    if not isinstance(chunk_id, str) or not chunk_id:
        raise RetrievalError("retriever returned evidence without chunk_id metadata")
    source_line = metadata.get("source_line")
    if source_line is not None:
        try:
            source_line = int(source_line)
        except (TypeError, ValueError) as error:
            raise RetrievalError(f"invalid source_line for {chunk_id!r}") from error
    score = metadata.get("score")
    if score is not None:
        try:
            score = float(score)
        except (TypeError, ValueError) as error:
            raise RetrievalError(f"invalid score for {chunk_id!r}") from error
    references = metadata.get("graph_references") or []
    if not isinstance(references, list) or not all(isinstance(item, str) for item in references):
        raise RetrievalError(f"invalid graph references for {chunk_id!r}")
    return EvidenceChunk(
        chunk_id=chunk_id,
        text=content,
        document_id=metadata.get("document_id"),
        source_file=metadata.get("source_file"),
        source_line=source_line,
        score=score,
        graph_references=references,
        paragraph_hash=metadata.get("paragraph_hash"),
    )


def retrieve(
    request: QueryRequest,
    retriever: Any,
    *,
    retrieval_settings: RetrievalSettings | None = None,
) -> RetrievalResult:
    """Retrieve bounded, structured graph evidence for a validated request.

    Args:
        request: Validated question retaining its original trace ID.
        retriever: Neo4j GraphRAG retriever with a ``search`` method.
        retrieval_settings: Top-K and context bounds.

    Returns:
        A result retaining the original question, trace, mode, and evidence.

    Raises:
        RetrievalError: If the provider fails or returns malformed evidence.
    """
    settings = retrieval_settings or RetrievalSettings()
    try:
        raw_result = retriever.search(query_text=request.question, top_k=settings.top_k)
    except Exception as error:  # noqa: BLE001 - expose provider failure at app boundary
        raise RetrievalError(f"GraphRAG retrieval failed: {error}") from error
    items = getattr(raw_result, "items", None)
    if items is None:
        raise RetrievalError("GraphRAG retrieval response is missing items")
    evidence = [_evidence_from_item(item) for item in items[: settings.max_context_chunks]]
    return RetrievalResult(
        trace_id=request.trace_id,
        question=request.question,
        mode=settings.mode,
        evidence=evidence,
    )
