"""Deterministic tests for MuSiQue evidence correctness and artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from graphrag_pipeline.musique_verification import (
    compare_musique_evidence,
    verify_graph_evidence_membership,
    verify_query_correctness,
    write_run_artifact,
)
from graphrag_pipeline.query import EvidenceChunk, QueryRequest, QueryResponse
from graphrag_pipeline.retrieval import RetrievalResult


class DummyConnection:
    """Read-only fake returning graph rows for the requested chunk IDs."""

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, dict]] = []

    def run_read(self, query: str, parameters: dict | None = None) -> list[dict]:
        self.calls.append((query, parameters or {}))
        return self.rows


def request() -> QueryRequest:
    """Return a selected MuSiQue request."""
    return QueryRequest(
        question="Who acquired Northwind Labs?",
        trace_id="trace-correctness-1",
        origin="musique",
        record_id="record-1",
        source_file="records.jsonl",
        source_line=4,
    )


def raw_record() -> dict:
    """Return a labeled record used only by evaluation checks."""
    return {
        "id": "record-1",
        "question": "Who acquired Northwind Labs?",
        "paragraphs": [
            {"idx": 1, "title": "Deal", "paragraph_text": "Aster Systems acquired Northwind Labs."},
            {"idx": 2, "title": "Date", "paragraph_text": "The deal closed in 2021."},
        ],
        "supporting_facts": [["Deal", 1], ["Date", 2]],
        "answer": "Aster Systems",
    }


def evidence(*items: tuple[str, str]) -> RetrievalResult:
    """Build a retrieval result from chunk ID and text pairs."""
    return RetrievalResult(
        trace_id=request().trace_id,
        question=request().question,
        mode="vector_cypher",
        evidence=[EvidenceChunk(chunk_id=chunk_id, text=text) for chunk_id, text in items],
    )


def response(status: str = "succeeded") -> QueryResponse:
    """Build a response preserving the request trace and question."""
    return QueryResponse(
        status=status,
        trace_id=request().trace_id,
        question=request().question,
        origin="musique",
        record_id=request().record_id,
        source_file=request().source_file,
        source_line=request().source_line,
        answer="Aster Systems" if status == "succeeded" else None,
        retrieval_mode="vector_cypher",
        citations=["musique:record-1:paragraph:1"] if status == "succeeded" else [],
    )


def test_compare_musique_evidence_reports_supporting_paragraph_coverage() -> None:
    """Returned chunk IDs and text are compared with labeled paragraphs."""
    report = compare_musique_evidence(
        request(),
        evidence(
            ("musique:record-1:paragraph:1", "Aster Systems acquired Northwind Labs."),
            ("musique:record-1:paragraph:2", "The deal closed in 2021."),
        ),
        raw_record(),
    )

    assert report.matched_chunk_ids == [
        "musique:record-1:paragraph:1",
        "musique:record-1:paragraph:2",
    ]
    assert report.text_matched_chunk_ids == report.matched_chunk_ids
    assert report.supporting_paragraphs_covered is True


def test_compare_musique_evidence_marks_unknown_and_missing_supporting_chunks() -> None:
    """Incorrect IDs and incomplete supporting evidence remain visible."""
    report = compare_musique_evidence(
        request(),
        evidence(
            ("musique:record-1:paragraph:1", "Aster Systems acquired Northwind Labs."),
            ("other:chunk", "Unrelated text."),
        ),
        raw_record(),
    )

    assert report.unmatched_chunk_ids == ["other:chunk"]
    assert report.retrieved_supporting_paragraph_indices == [1]
    assert report.supporting_paragraphs_covered is False


def test_verify_graph_evidence_membership_requires_matching_graph_text() -> None:
    """A chunk is verified only when ID, record filter, and text all match."""
    connection = DummyConnection(
        [
            {
                "chunk_id": "musique:record-1:paragraph:1",
                "text": "Aster Systems acquired Northwind Labs.",
                "record_id": "record-1",
                "document_id": "document-1",
                "source_file": "records.jsonl",
                "source_line": 4,
            }
        ]
    )

    report = verify_graph_evidence_membership(
        connection,
        request(),
        list(evidence(("musique:record-1:paragraph:1", "Aster Systems acquired Northwind Labs.")).evidence),
    )

    assert report.status == "verified"
    assert report.verified_chunk_ids == ["musique:record-1:paragraph:1"]
    assert connection.calls[0][1]["record_id"] == "record-1"


def test_verify_query_correctness_marks_graph_not_checked_without_connection() -> None:
    """Offline correctness reports distinguish evaluation from graph checks."""
    retrieval = evidence(("musique:record-1:paragraph:1", "Aster Systems acquired Northwind Labs."))
    report = verify_query_correctness(request(), retrieval, response(), raw_record())

    assert report.trace_id == request().trace_id
    assert report.graph_membership.status == "not_checked"
    assert report.evidence.supporting_paragraphs_covered is False


def test_verify_query_correctness_rejects_response_trace_mismatch() -> None:
    """Artifacts cannot connect an answer to a different query trace."""
    retrieval = evidence(("musique:record-1:paragraph:1", "Aster Systems acquired Northwind Labs."))
    mismatched = response().model_copy(update={"trace_id": "other-trace"})

    with pytest.raises(ValueError, match="trace or question"):
        verify_query_correctness(request(), retrieval, mismatched, raw_record())


def test_write_run_artifact_contains_query_evidence_answer_and_configuration(tmp_path: Path) -> None:
    """Run artifacts retain the complete inspectable query-to-answer trace."""
    retrieval = evidence(("musique:record-1:paragraph:1", "Aster Systems acquired Northwind Labs."))
    correctness = verify_query_correctness(request(), retrieval, response(), raw_record())
    artifact_path = tmp_path / "runs" / "answer.json"

    write_run_artifact(
        artifact_path,
        request(),
        retrieval,
        response(),
        correctness,
        configuration={"retrieval_mode": "vector_cypher", "top_k": 5},
        timings={"retrieval_seconds": 0.25},
        errors=[],
    )

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["request"]["question"] == request().question
    assert artifact["retrieval"]["mode"] == "vector_cypher"
    assert artifact["response"]["citations"] == ["musique:record-1:paragraph:1"]
    assert artifact["correctness"]["graph_membership"]["status"] == "not_checked"
    assert artifact["configuration"]["top_k"] == 5
