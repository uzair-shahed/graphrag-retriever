"""Tests for run_requests' per-record success tracking (no live services).

Neo4j and SimpleKGPipeline construction are mocked at the module boundary so
these tests exercise the batching, progress, and success-counting logic
without opening a connection or calling OpenRouter.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import BaseModel
from graphrag_pipeline.extraction.schemas import ExtractionRequest, Provenance
from graphrag_pipeline.ingestion.load_musique_dataset import (
    count_fully_succeeded_records,
    run_requests,
)


def make_request(record_id: str, paragraph_idx: int) -> ExtractionRequest:
    """Build a minimal validated request for one record's paragraph."""
    return ExtractionRequest(
        chunk_id=f"musique:{record_id}:paragraph:{paragraph_idx}",
        record_id=record_id,
        document_id="musique:document:example",
        paragraph_idx=paragraph_idx,
        text="Example paragraph text.",
        title="Example",
        provenance=Provenance(
            source_dataset="musique",
            source_split="fixture",
            source_file="fixture.jsonl",
            source_sha256="abc123",
            source_line=1,
        ),
    )


@patch("graphrag_pipeline.ingestion.load_musique_dataset.build_simple_kg_pipeline")
@patch("graphrag_pipeline.ingestion.load_musique_dataset.Neo4jConnection")
def test_run_requests_counts_records_with_a_failing_chunk_as_not_succeeded(
    mock_connection_cls: MagicMock, mock_build_pipeline: MagicMock
) -> None:
    """A record with any failed chunk is excluded from the succeeded-record count."""
    mock_connection = MagicMock()
    mock_connection.connect.return_value = MagicMock()
    mock_connection_cls.return_value = mock_connection

    pipeline = MagicMock()
    pipeline.run_async = AsyncMock(
        side_effect=[
            SimpleNamespace(run_id="run-1"),
            SimpleNamespace(run_id="run-2"),
            RuntimeError("boom"),
            SimpleNamespace(run_id="run-4"),
        ]
    )
    mock_build_pipeline.return_value = pipeline

    requests = [
        make_request("record-1", 0),
        make_request("record-1", 1),
        make_request("record-2", 0),
        make_request("record-2", 1),
    ]

    succeeded, failed = asyncio.run(
        run_requests(requests, neo4j_settings=MagicMock(), openrouter_settings=MagicMock(), show_progress=False)
    )

    assert len(succeeded) == 3
    assert len(failed) == 1
    assert failed[0]["chunk_id"] == "musique:record-2:paragraph:0"

    records_loaded = count_fully_succeeded_records(requests, failed)
    assert records_loaded == 1


@patch("graphrag_pipeline.ingestion.load_musique_dataset.save_invalid_llm_response")
@patch("graphrag_pipeline.ingestion.load_musique_dataset.build_simple_kg_pipeline")
@patch("graphrag_pipeline.ingestion.load_musique_dataset.Neo4jConnection")
def test_run_requests_records_invalid_response_path(
    mock_connection_cls: MagicMock,
    mock_build_pipeline: MagicMock,
    mock_save_response: MagicMock,
) -> None:
    """A validation failure records the saved raw-response path for its chunk."""
    mock_connection = MagicMock()
    mock_connection.connect.return_value = MagicMock()
    mock_connection_cls.return_value = mock_connection

    class ExpectedGraph(BaseModel):
        nodes: list[dict]

    try:
        ExpectedGraph.model_validate_json('{"nodes": "invalid"}')
    except Exception as validation_error:
        pipeline = MagicMock()
        pipeline.run_async = AsyncMock(side_effect=validation_error)
        pipeline.runner.config.llm_config = {
            "default": SimpleNamespace(root=SimpleNamespace(last_response='{"nodes": "invalid"}'))
        }
        mock_build_pipeline.return_value = pipeline
        mock_save_response.return_value = Path("debugging_logs/invalid_llm_response.txt")

        _, failed = asyncio.run(
            run_requests(
                [make_request("record-1", 0)],
                neo4j_settings=MagicMock(),
                openrouter_settings=MagicMock(),
                show_progress=False,
            )
        )

    assert failed[0]["invalid_llm_response"] == "debugging_logs/invalid_llm_response.txt"
    mock_save_response.assert_called_once()


def test_count_fully_succeeded_records_with_no_failures() -> None:
    """Every attempted record counts as loaded when nothing failed."""
    requests = [make_request("record-1", 0), make_request("record-2", 0)]

    assert count_fully_succeeded_records(requests, failed=[]) == 2
