"""Deterministic tests for optional query-time MuSiQue preloading."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from graphrag_pipeline.ingestion.load_musique_dataset import load_selected_records
from graphrag_pipeline.query import QueryRequest, QueryResponse, run_query_batch, select_queries


def make_record(record_id: str, question: str) -> dict:
    """Build a valid MuSiQue record with one paragraph."""
    return {
        "id": record_id,
        "question": question,
        "paragraphs": [
            {"idx": 0, "title": "Evidence", "paragraph_text": "Evidence text."}
        ],
    }


def write_record(path: Path, record: dict) -> None:
    """Write one JSONL record."""
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def answer(request):
    """Return a deterministic successful answer response."""
    return QueryResponse(
        status="succeeded",
        trace_id=request.trace_id,
        question=request.question,
        origin=request.origin,
        record_id=request.record_id,
        source_file=request.source_file,
        source_line=request.source_line,
        preload=request.preload,
        answer="Grounded answer",
        retrieval_mode="fake",
        citations=["chunk-1"],
    )


def test_query_only_batch_does_not_call_preload_loader(tmp_path: Path) -> None:
    """The default query mode invokes QA without opening the ingestion path."""
    dataset = tmp_path / "records.jsonl"
    write_record(dataset, make_record("record-1", "Question one"))
    requests = select_queries([dataset])
    loader = AsyncMock()

    responses = asyncio.run(run_query_batch(requests, answer, preload_loader=loader))

    assert responses[0].status == "succeeded"
    loader.assert_not_awaited()


def test_preload_uses_selected_record_and_preserves_provenance(tmp_path: Path) -> None:
    """Preload planning adapts only the selected record's source line."""
    dataset = tmp_path / "records.jsonl"
    dataset.write_text(
        "\n" + json.dumps(make_record("record-1", "Question one")) + "\n"
        + json.dumps(make_record("record-2", "Question two")) + "\n",
        encoding="utf-8",
    )
    request = select_queries([dataset], record_id="record-2", preload=True)[0]

    summary = asyncio.run(load_selected_records([request], dry_run=True))

    assert summary.succeeded
    assert summary.records_read == 1
    assert summary.chunks_planned == 1
    assert summary.requested_records[0]["record_id"] == "record-2"
    assert summary.requested_records[0]["source_line"] == 3


async def fake_run_requests(requests, *_args, store=None, **_kwargs):
    """Stand in for the runner: persist every chunk as verified via the store."""
    store.register(requests)
    for request in requests:
        store.for_request(request).mark_verified(request.chunk_id)
    return [{"chunk_id": "chunk-1"}], []


def test_live_selected_preload_reuses_existing_request_runner(tmp_path: Path) -> None:
    """Live selected loading delegates chunks to the established runner."""
    dataset = tmp_path / "records.jsonl"
    write_record(dataset, make_record("record-1", "Question one"))
    request = select_queries([dataset], record_id="record-1", preload=True)[0]

    with patch(
        "graphrag_pipeline.ingestion.load_musique_dataset.run_requests",
        new=AsyncMock(side_effect=fake_run_requests),
    ) as run_requests:
        summary = asyncio.run(
            load_selected_records(
                [request],
                neo4j_settings=SimpleNamespace(),
                openrouter_settings=SimpleNamespace(),
                show_progress=False,
                state_dir=tmp_path / "state",
            )
        )

    assert summary.succeeded
    assert summary.records_succeeded == 1
    run_requests.assert_awaited_once()
    assert run_requests.await_args.args[0][0].record_id == "record-1"


def test_successful_preload_runs_answer_query() -> None:
    """A successful preload allows the answer callback to execute."""
    selected = QueryRequest(
        question="Question",
        trace_id="trace-1",
        origin="musique",
        record_id="record-1",
        source_file="records.jsonl",
        source_line=1,
        preload=True,
    )
    loader = AsyncMock(
        return_value=SimpleNamespace(succeeded=True, to_dict=lambda: {})
    )
    answer_query = AsyncMock(side_effect=answer)

    responses = asyncio.run(
        run_query_batch([selected], answer_query, preload_loader=loader)
    )

    assert responses[0].status == "succeeded"
    loader.assert_awaited_once()
    answer_query.assert_awaited_once_with(selected)


def test_preload_failure_prevents_answer_query() -> None:
    """A failed preload becomes a failed response without invoking QA."""
    selected = QueryRequest(
        question="Question",
        trace_id="trace-1",
        origin="musique",
        record_id="record-1",
        source_file="records.jsonl",
        source_line=1,
        preload=True,
    )
    loader = AsyncMock(
        return_value=SimpleNamespace(
            succeeded=False,
            to_dict=lambda: {"dead_lettered_records": [{"record_id": "record-1"}]},
        )
    )
    answer_query = AsyncMock()

    responses = asyncio.run(
        run_query_batch([selected], answer_query, preload_loader=loader)
    )

    assert responses[0].status == "failed"
    assert responses[0].preload is True
    answer_query.assert_not_awaited()
    loader.assert_awaited_once()
