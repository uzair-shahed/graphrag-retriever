"""Neo4j GraphRAG retrieval adapter for structured graph evidence.

Last modified: 2026-10-17

Purpose of this file:
    Construct a Neo4j ``VectorCypherRetriever``, submit a validated query, and
    map returned graph records into the application's ``EvidenceChunk``
    contract, including each seed chunk's associated entity subgraph.

Files that use this file:
    The query command and retrieval-test program call ``retrieve``. Unit tests
    mock the Neo4j retriever and embedding boundaries.

Caveats:
    The persisted graph stores chunk provenance (``chunk_id``, ``record_id``,
    ``document_id``, ``title``, ``source_file``) on the ``Document`` node, one
    hop from the vector-matched ``Chunk`` node via ``FROM_DOCUMENT`` -- the
    ``Chunk`` node itself only has ``text``/``embedding``/``index``. Both
    retrieval modes resolve provenance through that hop; a plain
    ``VectorRetriever`` cannot, so this module always builds a
    ``VectorCypherRetriever``. GraphRAG mode additionally traverses the seed
    chunk's own associated entity subgraph via ``apoc.path.subgraphAll``
    (entities connected through ``FROM_CHUNK``, plus entity-entity
    relationships); live inventory on 2026-10-17 confirmed the persisted graph
    has no cross-chunk or cross-record entity links, so traversal is
    intentionally scoped to one seed chunk's own subgraph, not a multi-chunk
    component. GraphRAG results are normalized into the same evidence contract
    as the VectorRAG path. GraphRAG-generated answers are retained so the
    answer boundary does not invoke the provider a second time; note that
    Neo4j's managed ``GraphRAG.search()`` prompt only sees chunk ``text``, not
    the structured ``graph_nodes``/``graph_relationships`` metadata -- that
    metadata is retained for evidence/citation/verification purposes even
    though it does not influence the managed answer's wording.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import neo4j
from neo4j_graphrag.embeddings.ollama import OllamaEmbeddings
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.generation.graphrag import GraphRAG
from neo4j_graphrag.generation.prompts import RagTemplate
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.types import RetrieverResultItem

from graphrag_pipeline.answer_context import (
    ANSWER_PROMPT_TEMPLATE,
    ANSWER_SYSTEM_INSTRUCTIONS,
    render_evidence_context,
)
from graphrag_pipeline.config import OllamaSettings, OpenRouterSettings
from graphrag_pipeline.query import EvidenceChunk, GraphNode, GraphRelationship, QueryRequest
from graphrag_pipeline.retrieval_prerequisites import VectorIndexSettings


class RetrievalError(RuntimeError):
    """Raised when Neo4j returns an unusable retrieval result or provider error."""


class RecordFilterRetriever(VectorCypherRetriever):
    """``VectorCypherRetriever`` that only filters through the vector index.

    Neo4j 2026.x lists filterable properties after the vector property in the
    index's ``properties`` (not in ``vector.filterable_properties``, which
    neo4j-graphrag 1.21 reads), so the stock retriever never sees them and falls
    back to a procedure path that is broken for ``VectorCypherRetriever``. This
    subclass reads them from ``properties`` and rejects any filter on a property
    the index does not declare, so a record constraint is never applied through
    an unreported fallback.
    """

    def _fetch_index_infos(self, vector_index_name: str) -> None:
        """Load index metadata and add the index's filterable properties."""
        super()._fetch_index_infos(vector_index_name)
        records = self.driver.execute_query(
            "SHOW VECTOR INDEXES YIELD name, properties WHERE name = $index_name RETURN properties",
            {"index_name": vector_index_name},
            database_=self.neo4j_database,
            routing_=neo4j.RoutingControl.READ,
        ).records
        indexed = list(records[0]["properties"]) if records else []
        self._filterable_properties = sorted({*self._filterable_properties, *indexed[1:]})

    def get_search_results(
        self,
        query_vector: list[float] | None = None,
        query_text: str | None = None,
        top_k: int = 5,
        effective_search_ratio: int = 1,
        query_params: dict[str, Any] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> Any:
        """Run vector search, refusing filters on properties the index lacks."""
        if filters:
            missing = sorted(set(filters) - set(self._filterable_properties))
            if missing:
                raise RetrievalError(
                    f"vector index {self.index_name!r} does not declare {missing} as filterable; "
                    "run `python -m graphrag_pipeline.vector_index_setup --apply`"
                )
        return super().get_search_results(
            query_vector=query_vector,
            query_text=query_text,
            top_k=top_k,
            effective_search_ratio=effective_search_ratio,
            query_params=query_params,
            filters=filters,
        )


# Relationship types that are lexical-graph/provenance infrastructure, not
# domain knowledge; traversal must never cross these to leave the seed chunk.
_STRUCTURAL_RELATIONSHIP_TYPES = frozenset({"FROM_DOCUMENT", "FROM_CHUNK", "NEXT_CHUNK"})
# Relationship type names are LLM-extracted (schema="FREE"); only identifiers
# matching this pattern are safe to compose into an APOC relationshipFilter.
_SAFE_RELATIONSHIP_TYPE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


RetrievalBackend = Literal["graphrag", "vectorrag"]


@dataclass(frozen=True)
class RetrievalSettings:
    """Bounded configuration for one GraphRAG retrieval call."""

    backend: RetrievalBackend = "vectorrag"
    mode: Literal["vector", "vector_cypher"] = "vector"
    top_k: int = 5
    max_context_chunks: int = 5
    max_graph_nodes: int = 200

    @property
    def candidate_k(self) -> int:
        """Return an over-fetched candidate bound for duplicate graph rows."""
        return self.top_k * 4

    @property
    def graphrag_seed_k(self) -> int:
        """Return the seed count for the GraphRAG backend.

        The managed ``GraphRAG.search()`` answers from every row it retrieves,
        so it must never be over-fetched beyond the retained evidence bound.
        """
        return min(self.top_k, self.max_context_chunks)

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.max_context_chunks < 1:
            raise ValueError("max_context_chunks must be positive")
        if self.max_graph_nodes < 1:
            raise ValueError("max_graph_nodes must be positive")
        if self.backend == "graphrag" and self.mode != "vector_cypher":
            raise ValueError("graphrag backend requires vector_cypher retrieval mode")


@dataclass(frozen=True)
class RetrievalResult:
    """Evidence returned for one query, retaining its question and trace."""

    trace_id: str
    question: str
    mode: str
    evidence: list[EvidenceChunk]
    backend: RetrievalBackend = "vectorrag"
    generated_answer: str | None = None

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
    """Map a raw Neo4j record to text plus provenance and graph metadata."""
    node = _record_value(record, "node")
    if isinstance(node, dict):
        values = node
    else:
        values = {
            key: _record_value(record, key)
            for key in (
                "chunk_id",
                "text",
                "document_id",
                "record_id",
                "source_file",
                "source_line",
                "source_sha256",
                "paragraph_idx",
                "title",
                "graph_nodes",
                "graph_relationships",
                "graph_complete",
            )
        }
    text = values.get("text") or _record_value(record, "text")
    chunk_id = values.get("chunk_id") or _record_value(record, "chunk_id")
    if not isinstance(chunk_id, str) or not chunk_id:
        raise RetrievalError("retrieval result is missing chunk_id")
    if not isinstance(text, str) or not text:
        raise RetrievalError(f"retrieval result for {chunk_id!r} is missing text")
    metadata = dict(values)
    metadata.update(
        {
            "text": text,
            "score": _record_value(record, "score"),
            "graph_references": _record_value(record, "graph_references") or [],
        }
    )
    # ``content`` is what GraphRAG.search shows the answer model, so it carries
    # the same bracketed chunk ID and graph lines as the VectorRAG prompt.
    evidence = _evidence_from_item(RetrieverResultItem(content=text, metadata=metadata))
    return RetrieverResultItem(content=render_evidence_context([evidence]), metadata=metadata)


def _retrieval_query_vector() -> str:
    """Return the Document-hop-only projection for the VectorRAG baseline.

    The vector-matched ``Chunk`` node only has ``text``/``embedding``/
    ``index``; provenance (``chunk_id``, ``record_id``, ``document_id``,
    ``title``, ``source_file``) lives one hop away on ``Document`` via
    ``FROM_DOCUMENT``. This mode never traverses entities.
    """
    return (
        "MATCH (node)-[:FROM_DOCUMENT]->(doc:Document) "
        "RETURN node.text AS text, doc.chunk_id AS chunk_id, "
        "doc.document_id AS document_id, doc.record_id AS record_id, "
        "doc.source_file AS source_file, doc.source_line AS source_line, "
        "doc.source_sha256 AS source_sha256, doc.paragraph_idx AS paragraph_idx, "
        "doc.title AS title, score"
    )


def _retrieval_query_vector_cypher(relationship_filter: str) -> str:
    """Return the Document-hop plus seed chunk entity-subgraph projection.

    Traverses only the seed chunk's own associated entities (via the reverse
    ``FROM_CHUNK`` relationship and the given domain-relationship allowlist)
    with no hop-depth ceiling; ``$node_limit``/``$query_limit`` are resource
    safeguards supplied as query parameters by ``retrieve``. ``FROM_DOCUMENT``
    is excluded from ``relationship_filter`` so traversal can never leave the
    seed chunk's own subgraph.
    """
    return (
        "MATCH (node)-[:FROM_DOCUMENT]->(doc:Document) "
        "CALL apoc.path.subgraphAll(node, {"
        f"relationshipFilter: '{relationship_filter}', "
        "minLevel: 0, maxLevel: -1, limit: $query_limit"
        "}) YIELD nodes, relationships "
        "RETURN node.text AS text, doc.chunk_id AS chunk_id, "
        "doc.document_id AS document_id, doc.record_id AS record_id, "
        "doc.source_file AS source_file, doc.source_line AS source_line, "
        "doc.source_sha256 AS source_sha256, doc.paragraph_idx AS paragraph_idx, "
        "doc.title AS title, score, "
        "[n IN nodes WHERE n:`__Entity__` | {"
        "id: elementId(n), "
        "labels: [l IN labels(n) WHERE NOT l IN ['__Entity__', '__KGBuilder__']], "
        "name: n.name}] AS graph_nodes, "
        "[r IN relationships WHERE type(r) <> 'FROM_CHUNK' | {"
        "type: type(r), start_node_id: elementId(startNode(r)), end_node_id: elementId(endNode(r))"
        "}] AS graph_relationships, "
        "(size(nodes) <= $node_limit) AS graph_complete"
    )


def _discover_relationship_allowlist(driver: Any, neo4j_database: str) -> str:
    """Build an APOC relationshipFilter from the live graph's relationship types.

    Returns ``FROM_CHUNK<`` (incoming, to reach a chunk's entities) plus every
    other observed relationship type, excluding lexical-graph infrastructure
    types (``FROM_DOCUMENT``/``NEXT_CHUNK``) so traversal cannot leave the
    seed chunk's own subgraph, and excluding any type name that is not a safe
    Cypher identifier (relationship types are LLM-extracted, schema="FREE").
    """
    try:
        with driver.session(database=neo4j_database) as session:
            rows = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType").data()
    except Exception as error:  # noqa: BLE001 - normalize driver errors at this boundary
        raise RetrievalError(f"relationship-type inventory failed: {error}") from error
    domain_types = sorted(
        {
            row["relationshipType"]
            for row in rows
            if row.get("relationshipType") not in _STRUCTURAL_RELATIONSHIP_TYPES
            and _SAFE_RELATIONSHIP_TYPE.match(row.get("relationshipType") or "")
        }
    )
    return "|".join(["FROM_CHUNK<", *domain_types])


def build_retriever(
    driver: Any,
    neo4j_database: str,
    model_settings: OpenRouterSettings | OllamaSettings,
    *,
    index_settings: VectorIndexSettings | None = None,
    retrieval_settings: RetrievalSettings | None = None,
    relationship_types: Sequence[str] | None = None,
) -> Any:
    """Build a configured Neo4j GraphRAG ``VectorCypherRetriever``.

    Args:
        driver: Neo4j driver passed directly to the supported retriever.
        neo4j_database: Database containing the verified vector index.
        model_settings: Existing OpenRouter or Ollama embedding settings.
        index_settings: Expected vector-index contract.
        retrieval_settings: Retrieval mode and bounds.
        relationship_types: Override for the live relationship-type inventory
            (``vector_cypher`` mode only); discovered from ``driver`` when
            omitted. Deterministic tests pass an explicit list.

    Returns:
        A configured ``RecordFilterRetriever`` (a ``VectorCypherRetriever``).
        A plain ``VectorRetriever``
        cannot be used: the vector-matched ``Chunk`` node does not carry the
        ``chunk_id``/``record_id`` provenance this application requires.
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
        return RecordFilterRetriever(**common, retrieval_query=_retrieval_query_vector())
    if retrieval.mode == "vector_cypher":
        if relationship_types is None:
            relationship_filter = _discover_relationship_allowlist(driver, neo4j_database)
        else:
            domain_types = sorted(
                {
                    rel_type
                    for rel_type in relationship_types
                    if rel_type not in _STRUCTURAL_RELATIONSHIP_TYPES and _SAFE_RELATIONSHIP_TYPE.match(rel_type)
                }
            )
            relationship_filter = "|".join(["FROM_CHUNK<", *domain_types])
        return RecordFilterRetriever(
            **common,
            retrieval_query=_retrieval_query_vector_cypher(relationship_filter),
        )
    raise ValueError(f"unsupported retrieval mode: {retrieval.mode}")


def build_graphrag(retriever: Any, llm: Any) -> GraphRAG:
    """Build Neo4j's managed GraphRAG search boundary.

    Args:
        retriever: Configured Neo4j-supported vector retriever.
        llm: Configured Neo4j-supported answer LLM.

    Returns:
        A GraphRAG instance that owns retrieval, context construction, and
        answer generation, using the same instructions and JSON output
        contract as the VectorRAG prompt; ``generate_answer`` validates its
        output with the same parser and citation check.
    """
    return GraphRAG(
        retriever=retriever,
        llm=llm,
        prompt_template=RagTemplate(
            template=ANSWER_PROMPT_TEMPLATE,
            expected_inputs=["context", "query_text"],
            system_instructions=ANSWER_SYSTEM_INSTRUCTIONS,
        ),
    )


def _evidence_from_item(item: Any) -> EvidenceChunk:
    """Validate one GraphRAG retriever item as an application evidence chunk."""
    content = getattr(item, "content", None)
    metadata = getattr(item, "metadata", None) or {}
    # Rendered ``content`` may carry citation brackets and graph lines; the raw
    # chunk text travels in metadata so evidence text stays verifiable.
    text = metadata.get("text") or content
    if not isinstance(text, str) or not text:
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
    graph_nodes = _graph_nodes_from_metadata(metadata, chunk_id)
    graph_relationships = _graph_relationships_from_metadata(metadata, chunk_id)
    graph_complete = metadata.get("graph_complete")
    if graph_complete is None:
        graph_complete = True
    elif not isinstance(graph_complete, bool):
        raise RetrievalError(f"invalid graph_complete flag for {chunk_id!r}")
    return EvidenceChunk(
        chunk_id=chunk_id,
        text=text,
        document_id=metadata.get("document_id"),
        source_file=metadata.get("source_file"),
        source_line=source_line,
        score=score,
        graph_references=references,
        graph_nodes=graph_nodes,
        graph_relationships=graph_relationships,
        graph_component_complete=graph_complete,
    )


def _graph_nodes_from_metadata(metadata: dict[str, Any], chunk_id: str) -> list[GraphNode]:
    """Validate the APOC-projected entity list for one seed chunk."""
    raw_nodes = metadata.get("graph_nodes") or []
    if not isinstance(raw_nodes, list):
        raise RetrievalError(f"invalid graph_nodes for {chunk_id!r}")
    nodes: list[GraphNode] = []
    for raw_node in raw_nodes:
        if not isinstance(raw_node, dict) or not isinstance(raw_node.get("id"), str) or not raw_node["id"]:
            raise RetrievalError(f"invalid graph node entry for {chunk_id!r}")
        labels = raw_node.get("labels") or []
        if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
            raise RetrievalError(f"invalid graph node labels for {chunk_id!r}")
        nodes.append(GraphNode(node_id=raw_node["id"], labels=labels, name=raw_node.get("name")))
    return nodes


def _graph_relationships_from_metadata(metadata: dict[str, Any], chunk_id: str) -> list[GraphRelationship]:
    """Validate the APOC-projected relationship list for one seed chunk."""
    raw_rels = metadata.get("graph_relationships") or []
    if not isinstance(raw_rels, list):
        raise RetrievalError(f"invalid graph_relationships for {chunk_id!r}")
    relationships: list[GraphRelationship] = []
    for raw_rel in raw_rels:
        if (
            not isinstance(raw_rel, dict)
            or not isinstance(raw_rel.get("type"), str)
            or not raw_rel["type"]
            or not isinstance(raw_rel.get("start_node_id"), str)
            or not raw_rel["start_node_id"]
            or not isinstance(raw_rel.get("end_node_id"), str)
            or not raw_rel["end_node_id"]
        ):
            raise RetrievalError(f"invalid graph relationship entry for {chunk_id!r}")
        relationships.append(
            GraphRelationship(
                type=raw_rel["type"],
                start_node_id=raw_rel["start_node_id"],
                end_node_id=raw_rel["end_node_id"],
            )
        )
    return relationships


def _unique_items(items: list[Any]) -> list[Any]:
    """Remove duplicate graph rows while preserving vector-rank order."""
    unique: list[Any] = []
    seen_chunk_ids: set[str] = set()
    for item in items:
        metadata = getattr(item, "metadata", None) or {}
        chunk_id = metadata.get("chunk_id")
        if not isinstance(chunk_id, str):
            unique.append(item)
            continue
        if chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)
        unique.append(item)
    return unique


def _retrieval_query_text(request: QueryRequest) -> str:
    """Disambiguate MuSiQue retrieval with titles from the selected record."""
    if request.origin != "musique" or not request.paragraph_titles:
        return request.question
    unique_titles = list(dict.fromkeys(request.paragraph_titles))
    question_terms = {
        term.lower().strip("?.,:;()")
        for term in request.question.split()
        if len(term.strip("?.,:;()")) >= 4
    }
    specific_titles = [
        title
        for title in unique_titles
        if "(" in title
        and any(term in title.lower() for term in question_terms)
    ]
    if not specific_titles:
        specific_titles = [title for title in unique_titles if "(" in title]
    titles = "; ".join(specific_titles[:3] or unique_titles[:3])
    return f"{request.question}\nRelevant source titles: {titles}"


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
    retrieval_query_text = _retrieval_query_text(request)
    graph_query_params = (
        {"node_limit": settings.max_graph_nodes, "query_limit": settings.max_graph_nodes + 1}
        if settings.mode == "vector_cypher"
        else None
    )
    try:
        if settings.backend == "graphrag":
            retriever_config: dict[str, Any] = {
                "top_k": settings.graphrag_seed_k,
            }
            if request.record_id:
                retriever_config["filters"] = {"record_id": request.record_id}
            if graph_query_params is not None:
                retriever_config["query_params"] = graph_query_params
            raw_result = retriever.search(
                query_text=retrieval_query_text,
                retriever_config=retriever_config,
                return_context=True,
            )
            retriever_result = getattr(raw_result, "retriever_result", None)
            items = getattr(retriever_result, "items", None)
            if items is None:
                raise RetrievalError("GraphRAG response is missing retriever context")
            if not items:
                return RetrievalResult(
                    trace_id=request.trace_id,
                    question=request.question,
                    mode="graphrag",
                    evidence=[],
                    backend="graphrag",
                )
            unique_items = _unique_items(items)
            if len(unique_items) > settings.max_context_chunks:
                # The answer was already generated from these rows, so trimming
                # evidence now would hide support the answer may depend on.
                raise RetrievalError(
                    f"GraphRAG answer context has {len(unique_items)} chunks, exceeding "
                    f"max_context_chunks={settings.max_context_chunks}"
                )
            evidence = [_evidence_from_item(item) for item in unique_items]
            generated_answer = getattr(raw_result, "answer", None)
            if not isinstance(generated_answer, str):
                raise RetrievalError("GraphRAG response is missing an answer")
            return RetrievalResult(
                trace_id=request.trace_id,
                question=request.question,
                mode="graphrag",
                evidence=evidence,
                backend="graphrag",
                generated_answer=generated_answer,
            )
        search_kwargs: dict[str, Any] = {"top_k": settings.candidate_k}
        if request.record_id:
            search_kwargs["filters"] = {"record_id": request.record_id}
        if graph_query_params is not None:
            search_kwargs["query_params"] = graph_query_params
        raw_result = retriever.search(query_text=retrieval_query_text, **search_kwargs)
    except Exception as error:  # noqa: BLE001 - expose provider failure at app boundary
        raise RetrievalError(f"GraphRAG retrieval failed: {error}") from error
    items = getattr(raw_result, "items", None)
    if items is None:
        raise RetrievalError("GraphRAG retrieval response is missing items")
    evidence = [
        _evidence_from_item(item)
        for item in _unique_items(items)[: settings.max_context_chunks]
    ]
    return RetrievalResult(
        trace_id=request.trace_id,
        question=request.question,
        mode=settings.mode,
        evidence=evidence,
        backend="vectorrag",
    )
