"""Deterministic tests for MuSiQue question-to-query selection."""

import json
from pathlib import Path

import pytest

from graphrag_pipeline.config import dataset_paths_from_env
from graphrag_pipeline.query import (
    QuerySelectionError,
    QueryResponse,
    custom_query,
    select_queries,
)


def write_records(path: Path, records: list[dict]) -> None:
    """Write deterministic JSONL records for selection tests."""
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def make_record(record_id: str, question: str) -> dict:
    """Build the minimum answer-bearing record shape."""
    return {
        "id": record_id,
        "question": question,
        "paragraphs": [{"idx": 0, "title": "Evidence", "paragraph_text": "Text."}],
    }


def test_select_queries_preserves_record_question_and_provenance(tmp_path: Path) -> None:
    """A record ID selects the exact question and source line unchanged."""
    dataset = tmp_path / "configured-records.jsonl"
    question = "Who is the spouse of the Green performer?"
    write_records(dataset, [make_record("record-1", question)])

    requests = select_queries([dataset], record_id="record-1", trace_id="trace-1")

    assert len(requests) == 1
    assert requests[0].question == question
    assert requests[0].record_id == "record-1"
    assert requests[0].source_file == str(dataset)
    assert requests[0].source_split == "configured-records"
    assert requests[0].source_line == 1
    assert requests[0].paragraph_indices == [0]
    assert requests[0].paragraph_titles == ["Evidence"]
    assert requests[0].trace_id == "trace-1"
    assert requests[0].preload is False


def test_select_queries_preserves_explicit_preload_mode(tmp_path: Path) -> None:
    """The preload flag is carried into every selected request."""
    dataset = tmp_path / "records.jsonl"
    write_records(dataset, [make_record("record-1", "Question one")])

    requests = select_queries([dataset], record_id="record-1", preload=True)

    assert requests[0].preload is True


def test_select_queries_without_record_id_returns_bounded_batch(tmp_path: Path) -> None:
    """No selector returns records from configured files in stable order."""
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_records(first, [make_record("record-1", "Question one")])
    write_records(second, [make_record("record-2", "Question two")])

    requests = select_queries([first, second], max_records=2, trace_id="trace-2")

    assert [request.record_id for request in requests] == ["record-1", "record-2"]
    assert [request.question for request in requests] == ["Question one", "Question two"]


def test_select_queries_rejects_missing_record_id(tmp_path: Path) -> None:
    """A requested record that is absent cannot become a different query."""
    dataset = tmp_path / "records.jsonl"
    write_records(dataset, [make_record("record-1", "Question one")])

    with pytest.raises(QuerySelectionError, match="record ID not found"):
        select_queries([dataset], record_id="missing")


def test_select_queries_rejects_duplicate_ids_across_files(tmp_path: Path) -> None:
    """Duplicate IDs are explicit errors rather than silently merged records."""
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    duplicate = make_record("record-1", "Different question")
    write_records(first, [make_record("record-1", "Question one")])
    write_records(second, [duplicate])

    with pytest.raises(QuerySelectionError, match="duplicate record ID"):
        select_queries([first, second])


def test_custom_query_has_no_false_dataset_provenance() -> None:
    """Custom questions remain distinguishable from MuSiQue questions."""
    request = custom_query("What is in the graph?", trace_id="trace-custom")

    assert request.origin == "custom"
    assert request.record_id is None
    assert request.source_file is None


def test_query_response_requires_evidence_for_successful_shape() -> None:
    """The response contract carries status, answer, evidence, and citations."""
    response = QueryResponse(
        status="succeeded",
        trace_id="trace-1",
        question="Question",
        answer="Answer",
        retrieval_mode="vector",
        citations=["chunk-1"],
    )

    assert response.answer == "Answer"
    assert response.citations == ["chunk-1"]


def test_query_response_preserves_request_provenance() -> None:
    """A response can retain the selected record and preload mode."""
    response = QueryResponse(
        status="succeeded",
        trace_id="trace-1",
        question="Question",
        origin="musique",
        record_id="record-1",
        source_file="configured-records.jsonl",
        source_line=1,
        preload=True,
        answer="Answer",
        retrieval_mode="vector",
        citations=["chunk-1"],
    )

    assert response.record_id == "record-1"
    assert response.preload is True


def test_dataset_paths_from_env_preserves_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configured dataset paths are parsed in deterministic order."""
    monkeypatch.setenv("GRAPH_RAG_DATASET_PATHS", "one.jsonl, second.jsonl")

    assert dataset_paths_from_env() == [Path("one.jsonl"), Path("second.jsonl")]


def test_dataset_paths_from_env_rejects_missing_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    """The query path fails clearly when no dataset source is configured."""
    monkeypatch.delenv("GRAPH_RAG_DATASET_PATHS", raising=False)

    with pytest.raises(ValueError, match="GRAPH_RAG_DATASET_PATHS"):
        dataset_paths_from_env()