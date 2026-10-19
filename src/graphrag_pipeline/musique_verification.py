"""MuSiQue-specific correctness checks and reproducible query artifacts.

Last modified: 2026-10-06

Purpose of this file:
    Compare graph-retrieved evidence with a selected MuSiQue record, report
    supporting-paragraph coverage as an evaluation signal, verify returned
    chunks against Neo4j read-only, and persist an inspectable run artifact.

Files that use this file:
    The query command and retrieval-test program can call
    ``verify_query_correctness`` after retrieval and answer generation. Unit
    tests use deterministic records, responses, and read-only connection fakes.

Caveats:
    Raw MuSiQue labels are accepted only for evaluation. They are never passed
    to retrieval or answer-generation prompts. Graph verification requires a
    connection exposing ``run_read``; without one, the report explicitly marks
    graph membership as not checked.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr

from graphrag_pipeline.query import EvidenceChunk, QueryRequest, QueryResponse
from graphrag_pipeline.retrieval import RetrievalResult

_CHUNK_PARAGRAPH = re.compile(r":paragraph:(?P<paragraph_idx>[0-9]+)$")


class EvidenceCorrectness(BaseModel):
    """Evidence-to-record comparison for one query."""

    model_config = ConfigDict(extra="forbid")

    checked_chunk_ids: list[StrictStr] = Field(default_factory=list)
    matched_chunk_ids: list[StrictStr] = Field(default_factory=list)
    unmatched_chunk_ids: list[StrictStr] = Field(default_factory=list)
    text_matched_chunk_ids: list[StrictStr] = Field(default_factory=list)
    supporting_paragraph_indices: list[StrictInt] = Field(default_factory=list)
    retrieved_supporting_paragraph_indices: list[StrictInt] = Field(default_factory=list)
    supporting_paragraphs_covered: StrictBool | None = None


class GraphMembership(BaseModel):
    """Read-only graph membership result for returned evidence."""

    model_config = ConfigDict(extra="forbid")

    status: StrictStr
    checked_chunk_ids: list[StrictStr] = Field(default_factory=list)
    verified_chunk_ids: list[StrictStr] = Field(default_factory=list)
    graph_trace: list[dict[str, Any]] = Field(default_factory=list)
    error: StrictStr | None = None


class CorrectnessReport(BaseModel):
    """Machine-readable MuSiQue correctness report."""

    model_config = ConfigDict(extra="forbid")

    trace_id: StrictStr
    question: StrictStr
    record_id: StrictStr | None = None
    evidence: EvidenceCorrectness
    graph_membership: GraphMembership


def _supporting_indices(raw_record: dict[str, Any]) -> set[int]:
    """Extract supporting paragraph indices without altering the raw record.

    In MuSiQue, supporting paragraphs are marked with is_supporting=True directly
    on each item in the paragraphs list (unlike HotpotQA's supporting_facts list).
    """
    paragraphs = raw_record.get("paragraphs", [])
    supporting: set[int] = set()
    for item in paragraphs:
        if isinstance(item, dict) and item.get("is_supporting") and "idx" in item:
            try:
                supporting.add(int(item["idx"]))
            except (TypeError, ValueError):
                continue
    if supporting:
        return supporting

    # Backward compatibility fallback for synthetic/legacy fixtures using supporting_facts
    values: set[int] = set()
    for item in raw_record.get("supporting_facts", []):
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                values.add(int(item[1]))
            except (TypeError, ValueError):
                continue
        elif isinstance(item, dict):
            index = item.get("idx", item.get("paragraph_idx"))
            try:
                if index is not None:
                    values.add(int(index))
            except (TypeError, ValueError):
                continue
    return values


def _paragraphs(raw_record: dict[str, Any]) -> dict[int, str]:
    """Return selected record paragraph text keyed by MuSiQue paragraph index."""
    result: dict[int, str] = {}
    for paragraph in raw_record.get("paragraphs", []):
        if not isinstance(paragraph, dict):
            continue
        index = paragraph.get("idx")
        text = paragraph.get("paragraph_text", paragraph.get("text"))
        if isinstance(index, int) and isinstance(text, str):
            result[index] = text
    return result


def _paragraph_index(chunk_id: str) -> int | None:
    """Read the adapter's paragraph index from a graph chunk ID."""
    match = _CHUNK_PARAGRAPH.search(chunk_id)
    return int(match.group("paragraph_idx")) if match else None


def compare_musique_evidence(
    request: QueryRequest,
    retrieval: RetrievalResult,
    raw_record: dict[str, Any],
) -> EvidenceCorrectness:
    """Compare retrieved chunk IDs and text with one selected MuSiQue record.

    Args:
        request: Validated request identifying the selected record.
        retrieval: Bounded graph evidence returned for the request.
        raw_record: Original JSONL record, including labels for evaluation only.

    Returns:
        Structured evidence and supporting-paragraph coverage results.

    Raises:
        ValueError: If a MuSiQue request does not match the supplied record.
    """
    record_id = raw_record.get("id")
    if request.origin == "musique" and record_id != request.record_id:
        raise ValueError("correctness record ID does not match query request")

    paragraphs = _paragraphs(raw_record)
    supporting = _supporting_indices(raw_record)
    checked: list[str] = []
    matched: list[str] = []
    text_matched: list[str] = []
    retrieved_indices: set[int] = set()

    for evidence in retrieval.evidence:
        checked.append(evidence.chunk_id)
        paragraph_idx = _paragraph_index(evidence.chunk_id)
        if paragraph_idx is None or paragraph_idx not in paragraphs:
            continue
        matched.append(evidence.chunk_id)
        if evidence.text in paragraphs[paragraph_idx] or paragraphs[paragraph_idx] in evidence.text:
            text_matched.append(evidence.chunk_id)
            retrieved_indices.add(paragraph_idx)

    supporting_coverage: bool | None = None
    if supporting:
        supporting_coverage = supporting.issubset(retrieved_indices)

    return EvidenceCorrectness(
        checked_chunk_ids=checked,
        matched_chunk_ids=matched,
        unmatched_chunk_ids=[chunk_id for chunk_id in checked if chunk_id not in matched],
        text_matched_chunk_ids=text_matched,
        supporting_paragraph_indices=sorted(supporting),
        retrieved_supporting_paragraph_indices=sorted(supporting & retrieved_indices),
        supporting_paragraphs_covered=supporting_coverage,
    )


def verify_graph_evidence_membership(
    connection: Any,
    request: QueryRequest,
    evidence: list[EvidenceChunk],
) -> GraphMembership:
    """Verify returned chunks and source text in Neo4j using read-only queries.

    Args:
        connection: Read-only connection exposing ``run_read``.
        request: Request whose record ID constrains graph membership.
        evidence: Chunks returned by Neo4j retrieval.

    Returns:
        Verified chunk IDs plus graph provenance rows and relationship traces.

    Raises:
        ValueError: If a MuSiQue request lacks a record ID.
    """
    chunk_ids = [item.chunk_id for item in evidence]
    if not chunk_ids:
        return GraphMembership(status="empty", checked_chunk_ids=[])
    if request.origin == "musique" and not request.record_id:
        raise ValueError("MuSiQue graph verification requires record_id")

    try:
        rows = connection.run_read(
            """
            MATCH (c:Chunk)
            WHERE c.chunk_id IN $chunk_ids
              AND ($record_id IS NULL OR c.record_id = $record_id)
            RETURN c.chunk_id AS chunk_id, c.text AS text,
                   c.record_id AS record_id, c.document_id AS document_id,
                   c.source_file AS source_file, c.source_line AS source_line
            ORDER BY c.chunk_id
            """,
            {"chunk_ids": chunk_ids, "record_id": request.record_id},
        )
        row_by_id = {row.get("chunk_id"): row for row in rows}
        verified: list[str] = []
        trace: list[dict[str, Any]] = []
        for item in evidence:
            row = row_by_id.get(item.chunk_id)
            if row is None:
                continue
            if row.get("text") != item.text:
                continue
            verified.append(item.chunk_id)
            trace.append(row)
        status = "verified" if len(verified) == len(chunk_ids) else "incomplete"
        return GraphMembership(
            status=status,
            checked_chunk_ids=chunk_ids,
            verified_chunk_ids=verified,
            graph_trace=trace,
        )
    except Exception as error:  # noqa: BLE001 - report graph verification at this boundary
        return GraphMembership(
            status="failed",
            checked_chunk_ids=chunk_ids,
            error=f"graph evidence verification failed: {error}",
        )


def verify_query_correctness(
    request: QueryRequest,
    retrieval: RetrievalResult,
    response: QueryResponse,
    raw_record: dict[str, Any] | None = None,
    *,
    connection: Any | None = None,
) -> CorrectnessReport:
    """Compose MuSiQue evidence comparison and optional graph membership checks.

    Args:
        request: Original validated query request.
        retrieval: Graph retrieval result used for answer generation.
        response: Final answer or insufficient-evidence response.
        raw_record: Original MuSiQue record when evaluating a dataset query.
        connection: Optional read-only Neo4j connection.

    Returns:
        A report linking trace, question, evidence, answer status, and graph trace.
    """
    if raw_record is None:
        evidence_report = EvidenceCorrectness(checked_chunk_ids=[item.chunk_id for item in retrieval.evidence])
    else:
        evidence_report = compare_musique_evidence(request, retrieval, raw_record)
    graph_report = (
        verify_graph_evidence_membership(connection, request, retrieval.evidence)
        if connection is not None
        else GraphMembership(status="not_checked", checked_chunk_ids=[item.chunk_id for item in retrieval.evidence])
    )
    # Validate the response is tied to the same question and trace before writing an artifact.
    if response.trace_id != request.trace_id or response.question != request.question:
        raise ValueError("answer response does not preserve query trace or question")
    return CorrectnessReport(
        trace_id=request.trace_id,
        question=request.question,
        record_id=request.record_id,
        evidence=evidence_report,
        graph_membership=graph_report,
    )


def write_run_artifact(
    path: Path,
    request: QueryRequest,
    retrieval: RetrievalResult,
    response: QueryResponse,
    correctness: CorrectnessReport,
    *,
    configuration: dict[str, Any] | None = None,
    timings: dict[str, float] | None = None,
    errors: list[str] | None = None,
) -> None:
    """Write a deterministic JSON artifact containing the complete QA trace.

    Args:
        path: Output JSON path; parent directories are created as needed.
        request: Original question and provenance.
        retrieval: Retrieved evidence and mode.
        response: Final answer status, citations, and error.
        correctness: MuSiQue and graph verification report.
        configuration: Provider/index settings safe to record.
        timings: Named elapsed durations in seconds.
        errors: Additional redacted errors to retain.

    Returns:
        None. The artifact is written atomically by completing one JSON write.
    """
    retrieval_payload = {
        "trace_id": retrieval.trace_id,
        "question": retrieval.question,
        "mode": retrieval.mode,
        "evidence": [item.model_dump(mode="json") for item in retrieval.evidence],
    }
    artifact = {
        "request": request.model_dump(mode="json"),
        "retrieval": retrieval_payload,
        "response": response.model_dump(mode="json"),
        "correctness": correctness.model_dump(mode="json"),
        "configuration": configuration or {},
        "timings": timings or {},
        "errors": errors or [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
