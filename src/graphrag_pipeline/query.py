"""MuSiQue question-selection and question-to-answer contracts.

Last modified: 2026-10-06

Purpose of this file:
    Select MuSiQue questions from configured JSONL files and validate the
    request/response boundary used by the future GraphRAG query path.

Files that use this file:
    The future retrieval and answer command will consume ``select_queries``
    and return ``QueryResponse`` objects. Unit tests cover deterministic
    selection and contract validation.

Caveats:
    This module does not connect to Neo4j, retrieve evidence, or generate
    answers. It intentionally freezes the data contract before those steps
    are implemented.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Awaitable, Callable
from inspect import isawaitable
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

from graphrag_pipeline.ingestion.record_adapter import RawMusiqueRecord


class QuerySelectionError(ValueError):
    """Raised when configured dataset files cannot produce valid query inputs."""


class QueryRequest(BaseModel):
    """Validated question submitted to the GraphRAG query path.

    Args:
        question: Exact question text sent to retrieval and answer generation.
        trace_id: Identifier connecting selection, retrieval, and response.
        origin: Whether the question came from MuSiQue or was custom supplied.
        record_id: MuSiQue ID when the question came from a dataset record.
        source_split: Dataset split label when the question came from a file.
        source_file: Dataset path when the question came from a file.
        source_line: One-based JSONL line when the question came from a file.
        paragraph_indices: Paragraph indices from the selected record.
        paragraph_titles: Paragraph titles from the selected record.
        preload: Whether the selected record must be loaded before QA.

    Returns:
        A validated request object preserving question origin and provenance.
    """

    model_config = ConfigDict(extra="forbid")

    question: StrictStr = Field(min_length=1)
    trace_id: StrictStr = Field(min_length=1)
    origin: Literal["musique", "custom"]
    record_id: StrictStr | None = None
    source_split: StrictStr | None = None
    source_file: StrictStr | None = None
    source_line: StrictInt | None = Field(default=None, ge=1)
    paragraph_indices: list[StrictInt] = Field(default_factory=list)
    paragraph_titles: list[StrictStr] = Field(default_factory=list)
    preload: StrictBool = False
    retrieval_backend: Literal["graphrag", "vectorrag"] = "vectorrag"


class GraphNode(BaseModel):
    """One extracted entity node in a chunk's associated knowledge subgraph."""

    model_config = ConfigDict(extra="forbid")

    node_id: StrictStr = Field(min_length=1)
    labels: list[StrictStr] = Field(default_factory=list)
    name: StrictStr | None = None


class GraphRelationship(BaseModel):
    """One relationship between two nodes in a chunk's associated subgraph."""

    model_config = ConfigDict(extra="forbid")

    type: StrictStr = Field(min_length=1)
    start_node_id: StrictStr = Field(min_length=1)
    end_node_id: StrictStr = Field(min_length=1)


class EvidenceChunk(BaseModel):
    """One graph-backed evidence chunk returned by retrieval.

    Args:
        graph_nodes: Entities traversed from this chunk's seed node (GraphRAG
            mode only; always empty for the VectorRAG vector-only baseline).
        graph_relationships: Relationships among ``graph_nodes`` and the seed.
        graph_component_complete: Whether the returned subgraph is the full
            seed-associated component, or was truncated by a resource budget.
    """

    model_config = ConfigDict(extra="forbid")

    chunk_id: StrictStr = Field(min_length=1)
    text: StrictStr = Field(min_length=1)
    document_id: StrictStr | None = None
    source_file: StrictStr | None = None
    source_line: StrictInt | None = Field(default=None, ge=1)
    score: StrictFloat | None = None
    graph_references: list[StrictStr] = Field(default_factory=list)
    graph_nodes: list[GraphNode] = Field(default_factory=list)
    graph_relationships: list[GraphRelationship] = Field(default_factory=list)
    graph_component_complete: StrictBool = True


class QueryResponse(BaseModel):
    """Structured result returned by the GraphRAG question-answering path.

    Args:
        status: ``succeeded``, ``insufficient_evidence``, or ``failed``.
        trace_id: Identifier copied from the corresponding query request.
        question: Original submitted question.
        origin: Origin of the question, including MuSiQue provenance when present.
        record_id: Source MuSiQue record ID when applicable.
        source_file: Source dataset file when applicable.
        source_line: Source JSONL line when applicable.
        preload: Whether record preloading was requested for this result.
        answer: Generated answer when available.
        retrieval_mode: Configured GraphRAG retrieval mode.
        evidence: Graph-backed chunks used by the answer path.
        citations: Evidence IDs or source references supporting the answer.
        latency_seconds: End-to-end query latency when measured.
        error: Redacted failure details when the query does not succeed.

    Returns:
        A validated, provenance-preserving query result.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["succeeded", "insufficient_evidence", "failed"]
    trace_id: StrictStr = Field(min_length=1)
    question: StrictStr = Field(min_length=1)
    origin: Literal["musique", "custom"] = "custom"
    record_id: StrictStr | None = None
    source_file: StrictStr | None = None
    source_line: StrictInt | None = Field(default=None, ge=1)
    preload: StrictBool = False
    retrieval_backend: Literal["graphrag", "vectorrag"] = "vectorrag"
    answer: StrictStr | None = None
    retrieval_mode: StrictStr = Field(min_length=1)
    evidence: list[EvidenceChunk] = Field(default_factory=list)
    citations: list[StrictStr] = Field(default_factory=list)
    latency_seconds: StrictFloat | None = Field(default=None, ge=0)
    error: StrictStr | None = None

    @model_validator(mode="after")
    def validate_success_contract(self) -> "QueryResponse":
        """Require an answer and citations whenever a response succeeds."""
        if self.status == "succeeded":
            if not self.answer or not self.answer.strip():
                raise ValueError("successful responses require a non-empty answer")
            if not self.citations:
                raise ValueError("successful responses require citations")
        if self.status == "insufficient_evidence" and (self.answer is not None or self.citations):
            raise ValueError("insufficient evidence responses cannot contain answers or citations")
        return self


def _source_split(path: Path) -> str:
    """Derive a stable split label from a MuSiQue dataset filename."""
    parts = path.stem.split("_")
    if len(parts) >= 4 and parts[0] == "musique":
        return f"{parts[1]}_{parts[-1]}"
    return path.stem


def _load_query_request(
    path: Path,
    source_line: int,
    payload: dict[str, Any],
    trace_id: str,
    *,
    preload: bool = False,
) -> QueryRequest:
    """Validate one JSONL record and convert its question to a request."""
    record_id = payload.get("id")
    question = payload.get("question")
    if not isinstance(record_id, str) or not record_id.strip():
        raise QuerySelectionError(f"{path}:{source_line}: missing record id")
    if not isinstance(question, str) or not question.strip():
        raise QuerySelectionError(f"{path}:{source_line}: missing question")
    try:
        record = RawMusiqueRecord.model_validate(payload)
    except Exception as error:  # noqa: BLE001 - normalize Pydantic details at this boundary
        raise QuerySelectionError(
            f"{path}:{source_line}: invalid MuSiQue record shape: {error}"
        ) from error
    return QueryRequest(
        question=question,
        trace_id=trace_id,
        origin="musique",
        record_id=record_id,
        source_split=_source_split(path),
        source_file=str(path),
        source_line=source_line,
        paragraph_indices=[paragraph.idx for paragraph in record.paragraphs],
        paragraph_titles=[paragraph.title for paragraph in record.paragraphs],
        preload=preload,
    )


def select_queries(
    dataset_paths: list[Path],
    *,
    record_id: str | None = None,
    max_records: int | None = None,
    trace_id: str | None = None,
    preload: bool = False,
) -> list[QueryRequest]:
    """Select validated MuSiQue questions from one or more JSONL files.

    Args:
        dataset_paths: JSONL files to search, in deterministic order.
        record_id: Optional exact ID; without it all records are selected.
        max_records: Optional global batch bound across the configured files.
        trace_id: Optional shared trace ID; generated when omitted.
        preload: Whether each selected record must be loaded before QA.

    Returns:
        Query requests preserving exact question text and source provenance.

    Raises:
        QuerySelectionError: If a file, record, JSON line, or duplicate ID is
            invalid, or if the requested record cannot be found.
        ValueError: If no dataset paths or a non-positive batch bound is given.
    """
    if not dataset_paths:
        raise ValueError("at least one dataset path is required")
    if max_records is not None and max_records < 1:
        raise ValueError("max_records must be positive")

    selected: list[QueryRequest] = []
    seen_ids: set[str] = set()
    resolved_trace_id = trace_id or str(uuid.uuid4())

    for path in dataset_paths:
        if not path.is_file():
            raise QuerySelectionError(f"dataset file does not exist: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for source_line, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                try:
                    payload = json.loads(raw_line)
                except json.JSONDecodeError as error:
                    raise QuerySelectionError(
                        f"{path}:{source_line}: invalid JSON: {error}"
                    ) from error
                if not isinstance(payload, dict):
                    raise QuerySelectionError(f"{path}:{source_line}: record must be an object")
                current_id = payload.get("id")
                if record_id is not None and current_id != record_id:
                    continue
                if not isinstance(current_id, str) or not current_id.strip():
                    raise QuerySelectionError(f"{path}:{source_line}: missing record id")
                if current_id in seen_ids:
                    raise QuerySelectionError(f"duplicate record ID across dataset files: {current_id}")
                request = _load_query_request(
                    path, source_line, payload, resolved_trace_id, preload=preload
                )
                selected.append(request)
                seen_ids.add(current_id)
                if max_records is not None and len(selected) >= max_records:
                    return selected

    if record_id is not None and not selected:
        raise QuerySelectionError(f"record ID not found: {record_id}")
    if not selected:
        raise QuerySelectionError("configured dataset files contain no selectable records")
    return selected


def custom_query(question: str, *, trace_id: str | None = None) -> QueryRequest:
    """Build a validated request for an externally supplied question.

    Args:
        question: Non-empty question not associated with a MuSiQue record.
        trace_id: Optional trace identifier, generated when omitted.

    Returns:
        A custom-origin query request with no false dataset provenance.
    """
    return QueryRequest(
        question=question,
        trace_id=trace_id or str(uuid.uuid4()),
        origin="custom",
    )


async def run_query_batch(
    requests: list[QueryRequest],
    answer_query: Callable[[QueryRequest], QueryResponse | Awaitable[QueryResponse]],
    *,
    preload_loader: Callable[..., Awaitable[Any]] | None = None,
    show_progress: bool = False,
) -> list[QueryResponse]:
    """Run QA requests with explicit, per-record optional preloading.

    Args:
        requests: Validated query requests in deterministic execution order.
        answer_query: Retrieval/answer callback invoked only after preload.
        preload_loader: Injectable selected-record loader used when requested.
        show_progress: Whether the shared ingestion path displays progress.

    Returns:
        One query response per request, including explicit preload failures.

    Raises:
        QuerySelectionError: If a custom question requests record preloading.
    """
    if preload_loader is None:
        from graphrag_pipeline.ingestion.load_musique_dataset import load_selected_records

        preload_loader = load_selected_records

    responses: list[QueryResponse] = []
    for request in requests:
        if request.preload:
            if request.origin != "musique" or not request.record_id:
                raise QuerySelectionError("preload requires a MuSiQue record-backed query")
            summary = await preload_loader([request], show_progress=show_progress)
            if not summary.succeeded:
                responses.append(
                    QueryResponse(
                        status="failed",
                        trace_id=request.trace_id,
                        question=request.question,
                        origin=request.origin,
                        record_id=request.record_id,
                        source_file=request.source_file,
                        source_line=request.source_line,
                        preload=True,
                        retrieval_mode="unavailable",
                        error=json.dumps(summary.to_dict(), sort_keys=True),
                    )
                )
                continue

        result = answer_query(request)
        responses.append(await result if isawaitable(result) else result)
    return responses
