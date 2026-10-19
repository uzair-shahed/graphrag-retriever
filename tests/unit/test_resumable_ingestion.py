"""Unit tests for resumable ingestion, completion markers, and uniqueness constraints."""

from unittest.mock import MagicMock

import pytest

from graphrag_pipeline.extraction.schemas import ExtractionRequest, Provenance
from graphrag_pipeline.ingestion.record_adapter import compute_paragraph_hash
from graphrag_pipeline.storage.resumable_ingestion import (
    PIPELINE_VERSION,
    SCHEMA_HASH,
    cleanup_all_incomplete_paragraphs,
    cleanup_incomplete_paragraph,
    ensure_paragraph_hash_constraint,
    load_completed_paragraph_hashes,
    mark_paragraph_complete,
)


def make_request(title: str, text: str, record_id: str = "rec_1", idx: int = 0) -> ExtractionRequest:
    p_hash = compute_paragraph_hash(title, text)
    return ExtractionRequest(
        chunk_id=f"musique:paragraph:{p_hash}",
        record_id=record_id,
        document_id=f"musique:document:{p_hash}",
        paragraph_idx=idx,
        paragraph_hash=p_hash,
        text=text,
        title=title,
        provenance=Provenance(
            source_dataset="musique",
            source_split="dev",
            source_file="dev.jsonl",
            source_sha256="sha",
            source_line=1,
        ),
    )


def test_ensure_paragraph_hash_constraint():
    mock_session = MagicMock()
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    ensure_paragraph_hash_constraint(mock_driver)

    mock_session.run.assert_called_once()
    query = mock_session.run.call_args[0][0]
    assert "CREATE CONSTRAINT paragraph_hash_unique IF NOT EXISTS" in query
    assert "FOR (d:Document) REQUIRE d.paragraph_hash IS UNIQUE" in query


def test_load_completed_paragraph_hashes():
    mock_session = MagicMock()
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    mock_session.run.return_value = [
        {"hash": "hash_1"},
        {"hash": "hash_2"},
        {"hash": None},
    ]

    completed = load_completed_paragraph_hashes(mock_driver)
    assert completed == {"hash_1", "hash_2"}
    query = mock_session.run.call_args[0][0]
    assert "MATCH (d:Document {ingest_status: 'complete'})" in query


def test_mark_paragraph_complete():
    mock_session = MagicMock()
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    mark_paragraph_complete(mock_driver, "test_hash")

    mock_session.run.assert_called_once()
    args, kwargs = mock_session.run.call_args
    assert "MATCH (d:Document {paragraph_hash: $h})" in args[0]
    assert "SET d.ingest_status = 'complete'" in args[0]
    params = args[1]
    assert params["h"] == "test_hash"
    assert params["v"] == PIPELINE_VERSION
    assert params["s"] == SCHEMA_HASH


def test_cleanup_incomplete_paragraph():
    mock_session = MagicMock()
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    mock_result = MagicMock()
    mock_result.single.return_value = {"deleted_count": 1}
    mock_session.run.return_value = mock_result

    count = cleanup_incomplete_paragraph(mock_driver, "partial_hash")
    assert count == 1
    args = mock_session.run.call_args[0]
    assert "WHERE d.ingest_status IS NULL OR d.ingest_status <> 'complete'" in args[0]
    assert "DETACH DELETE d, c, e" in args[0]


def test_cleanup_all_incomplete_paragraphs():
    mock_session = MagicMock()
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    mock_result = MagicMock()
    mock_result.single.return_value = {"deleted_count": 3}
    mock_session.run.return_value = mock_result

    count = cleanup_all_incomplete_paragraphs(mock_driver)
    assert count == 3
    args = mock_session.run.call_args[0]
    assert "DETACH DELETE d, c, e" in args[0]
