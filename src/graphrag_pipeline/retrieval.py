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

from neo4j import Driver
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
    raw_idx = metadata.get("paragraph_idx")
    paragraph_idx: int | None = None
    if raw_idx is not None:
        try:
            paragraph_idx = int(raw_idx)
        except (TypeError, ValueError):
            paragraph_idx = None

    return EvidenceChunk(
        chunk_id=chunk_id,
        text=content,
        document_id=metadata.get("document_id"),
        source_file=metadata.get("source_file"),
        source_line=source_line,
        score=score,
        graph_references=references,
        paragraph_hash=metadata.get("paragraph_hash"),
        paragraph_idx=paragraph_idx,
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


def retrieve_per_question(
    driver: Driver,
    embedder: Any,
    question_id: str,
    question_text: str,
    *,
    top_k: int = 5,
    trace_id: str | None = None,
    database: str = "neo4j",
) -> RetrievalResult:
    """Retrieve top-k candidate paragraphs for a question using exact cosine over IN_CONTEXT links.

    This implements MuSiQue's per-question retrieval contract:
    - Cosine similarity is computed directly on each candidate chunk embedding in the question context.
    - Zero leakage by construction: only documents linked via (q:Question {id: $qid})-[:IN_CONTEXT]->(d:Document)
      are ever evaluated or returned.

    Args:
        driver: Active Neo4j driver.
        embedder: Embedding model instance with ``embed_query`` or ``get_embeddings``.
        question_id: MuSiQue record ID identifying the Question node.
        question_text: Text of the question to embed.
        top_k: Number of highest-similarity evidence chunks to return.
        trace_id: Optional trace ID for provenance tracking.
        database: Target Neo4j database name.

    Returns:
        RetrievalResult containing the ranked EvidenceChunk list.
    """
    if hasattr(embedder, "embed_query"):
        qvec = embedder.embed_query(question_text)
    elif hasattr(embedder, "get_embeddings"):
        qvec = embedder.get_embeddings([question_text])[0]
    else:
        raise ValueError(f"Embedder {embedder!r} does not provide embed_query or get_embeddings")

    query = """
    MATCH (q:Question {id: $qid})-[ic:IN_CONTEXT]->(d:Document)<-[:FROM_DOCUMENT]-(c:Chunk)
    WHERE c.embedding IS NOT NULL
    WITH q, d, c, ic.idx AS idx, vector.similarity.cosine(c.embedding, $qvec) AS score
    ORDER BY score DESC
    LIMIT $top_k
    RETURN c.text AS text,
           d.paragraph_hash AS paragraph_hash,
           coalesce(d.title, c.title) AS title,
           d.document_id AS document_id,
           d.record_id AS record_id,
           d.source_file AS source_file,
           d.source_line AS source_line,
           idx,
           score
    """
    records, _, _ = driver.execute_query(
        query,
        qid=question_id,
        qvec=qvec,
        top_k=top_k,
        database_=database,
    )

    chunks: list[EvidenceChunk] = []
    for r in records:
        p_hash = r.get("paragraph_hash")
        chunk_id = f"musique:paragraph:{p_hash}" if p_hash else f"chunk-{r.get('idx')}"
        source_line = r.get("source_line")
        parsed_source_line: int | None = None
        if source_line is not None:
            try:
                parsed_source_line = int(source_line)
            except (TypeError, ValueError):
                parsed_source_line = None

        chunks.append(
            EvidenceChunk(
                chunk_id=chunk_id,
                text=r["text"],
                document_id=r.get("document_id"),
                source_file=r.get("source_file"),
                source_line=parsed_source_line,
                score=float(r["score"]) if r.get("score") is not None else None,
                graph_references=[],
                paragraph_hash=p_hash,
                paragraph_idx=int(r["idx"]) if r.get("idx") is not None else None,
            )
        )

    return RetrievalResult(
        trace_id=trace_id or f"trace-{question_id}",
        question=question_text,
        mode="per_question",
        evidence=chunks,
    )

