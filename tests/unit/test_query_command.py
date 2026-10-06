"""Deterministic command-level tests for the minimal QA command."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from graphrag_pipeline.query import custom_query, select_queries
from graphrag_pipeline.query_command import build_requests, execute_queries
from graphrag_pipeline.retrieval import RetrievalSettings


def record(record_id: str, question: str) -> dict:
    """Build a minimal MuSiQue record for command tests."""
    return {
        "id": record_id,
        "question": question,
        "paragraphs": [
            {"idx": 1, "title": "Evidence", "paragraph_text": "Aster Systems acquired Northwind Labs."}
        ],
        "supporting_facts": [["Evidence", 1]],
        "answer": "Aster Systems",
    }


def write_records(path: Path, records: list[dict]) -> None:
    """Write deterministic JSONL records."""
    path.write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")


def fake_retriever() -> MagicMock:
    """Return a retriever with one graph-backed chunk."""
    retriever = MagicMock()
    retriever.search.return_value = SimpleNamespace(
        items=[
            SimpleNamespace(
                content="Aster Systems acquired Northwind Labs.",
                metadata={
                    "chunk_id": "musique:record-1:paragraph:1",
                    "document_id": "document-1",
                    "source_file": "records.jsonl",
                    "source_line": 1,
                },
            )
        ]
    )
    return retriever


def fake_llm() -> MagicMock:
    """Return a provider that emits a valid cited answer."""
    llm = MagicMock()
    llm.invoke.return_value = SimpleNamespace(
        content='{"answer":"Aster Systems","citations":["musique:record-1:paragraph:1"]}'
    )
    return llm


def test_custom_question_command_preserves_external_origin(tmp_path: Path) -> None:
    """Custom questions execute without false MuSiQue provenance or preload."""
    request = custom_query("Who acquired Northwind Labs?", trace_id="custom-trace")
    loader = AsyncMock()

    responses = asyncio.run(
        execute_queries(
            [request],
            retriever=fake_retriever(),
            llm=fake_llm(),
            retrieval_settings=RetrievalSettings(),
            artifact_path=tmp_path / "custom.json",
            preload_loader=loader,
        )
    )

    assert responses[0].status == "succeeded"
    assert responses[0].origin == "custom"
    assert responses[0].record_id is None
    loader.assert_not_awaited()
    artifact = json.loads((tmp_path / "custom.json").read_text(encoding="utf-8"))
    assert artifact["request"]["trace_id"] == "custom-trace"


def test_record_id_command_routes_exact_question_and_writes_artifact(tmp_path: Path) -> None:
    """Record selection preserves the stored question and correctness metadata."""
    dataset = tmp_path / "records.jsonl"
    write_records(dataset, [record("record-1", "Stored question exactly")])
    requests = select_queries([dataset], record_id="record-1")

    responses = asyncio.run(
        execute_queries(
            requests,
            retriever=fake_retriever(),
            llm=fake_llm(),
            retrieval_settings=RetrievalSettings(),
            artifact_path=tmp_path / "record.json",
            dataset_paths=[dataset],
        )
    )

    assert responses[0].question == "Stored question exactly"
    artifact = json.loads((tmp_path / "record.json").read_text(encoding="utf-8"))
    assert artifact["request"]["record_id"] == "record-1"
    assert artifact["correctness"]["evidence"]["supporting_paragraphs_covered"] is True


def test_record_batch_artifacts_are_numbered(tmp_path: Path) -> None:
    """A bounded batch produces one inspectable artifact per query."""
    dataset = tmp_path / "records.jsonl"
    write_records(dataset, [record("record-1", "Question one"), record("record-2", "Question two")])
    requests = select_queries([dataset], max_records=2)

    responses = asyncio.run(
        execute_queries(
            requests,
            retriever=fake_retriever(),
            llm=fake_llm(),
            retrieval_settings=RetrievalSettings(),
            artifact_path=tmp_path / "batch.json",
            dataset_paths=[dataset],
        )
    )

    assert len(responses) == 2
    assert (tmp_path / "batch-1.json").is_file()
    assert (tmp_path / "batch-2.json").is_file()


def test_build_requests_rejects_question_with_dataset_selection(tmp_path: Path) -> None:
    """The CLI does not silently combine custom and MuSiQue selection modes."""
    args = SimpleNamespace(
        question="custom",
        record_id=None,
        dataset_path=[str(tmp_path / "records.jsonl")],
        max_records=None,
        preload=False,
    )

    try:
        build_requests(args)
    except ValueError as error:
        assert "cannot be combined" in str(error)
    else:
        raise AssertionError("expected mixed source modes to be rejected")
