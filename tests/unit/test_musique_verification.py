"""Deterministic tests for MuSiQue evidence correctness and artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from graphrag_pipeline.musique_verification import (
    _supporting_indices,
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


def test_supporting_indices_extracted_from_real_musique_fixtures() -> None:
    """Real MuSiQue records extract supporting indices directly from is_supporting flags."""
    # 2-hop record from MuSiQue Ans dev split: 2hop__460946_294723
    record_2hop = {
        "id": "2hop__460946_294723",
        "paragraphs": [
            {"idx": 0, "title": "Grant's First Stand", "is_supporting": False},
            {"idx": 5, "title": "Green performer bio", "is_supporting": True},
            {"idx": 7, "title": "Random doc", "is_supporting": False},
            {"idx": 10, "title": "Miquette Giraudy", "is_supporting": True},
        ],
    }
    assert _supporting_indices(record_2hop) == {5, 10}

    # 3-hop record from MuSiQue Ans dev split: 3hop1__454441_55349_651302
    record_3hop = {
        "id": "3hop1__454441_55349_651302",
        "paragraphs": [
            {"idx": 0, "title": "The Hobbit (South Park)", "is_supporting": True},
            {"idx": 1, "title": "Dog with a Blog", "is_supporting": False},
            {"idx": 7, "title": "Stan voice actor", "is_supporting": True},
            {"idx": 9, "title": "Trey Parker birthplace", "is_supporting": True},
        ],
    }
    assert _supporting_indices(record_3hop) == {0, 7, 9}

    # 4-hop record from MuSiQue Ans dev split: 4hop1__93963_170667_443779_52195
    record_4hop = {
        "id": "4hop1__93963_170667_443779_52195",
        "paragraphs": [
            {"idx": 1, "title": "WFGS", "is_supporting": False},
            {"idx": 3, "title": "Cathay Pacific Flight 780", "is_supporting": True},
            {"idx": 11, "title": "Lion Air hub airport", "is_supporting": True},
            {"idx": 13, "title": "Timor Leste Commission", "is_supporting": True},
            {"idx": 14, "title": "Francisco Guterres presidency", "is_supporting": True},
        ],
    }
    assert _supporting_indices(record_4hop) == {3, 11, 13, 14}


def test_inference_modules_do_not_import_evaluation_gold_labels() -> None:
    """Inference modules must not leak or import gold evaluation verification logic."""
    import ast

    inference_files = [
        Path("src/graphrag_pipeline/retrieval.py"),
        Path("src/graphrag_pipeline/answer.py"),
        Path("src/graphrag_pipeline/extraction/pipeline.py"),
        Path("src/graphrag_pipeline/query.py"),
        Path("src/graphrag_pipeline/ingestion/record_adapter.py"),
    ]
    forbidden_names = {"_supporting_indices", "compare_musique_evidence", "verify_query_correctness"}
    for file_path in inference_files:
        tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    assert alias.name not in forbidden_names, (
                        f"Inference module {file_path} forbidden from importing {alias.name}"
                    )


def test_musique_ans_dev_dataset_hop_count_matches_supporting_count() -> None:
    """All 2,417 Ans dev records must have supporting count equal to hop count."""
    ans_dev_path = Path("dataset/musique/data/data/musique_ans_v1.0_dev.jsonl")
    if not ans_dev_path.exists():
        pytest.skip("MuSiQue Ans dev dataset not downloaded locally.")

    total = 0
    with ans_dev_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            total += 1
            hop = int(record["id"].split("hop")[0])
            supporting = _supporting_indices(record)
            assert len(supporting) == hop, f"Record {record['id']} expected {hop} supporting, got {len(supporting)}"

    assert total == 2417

