"""Unit tests for read-only chunk reconciliation against Neo4j state."""

from __future__ import annotations

import pytest

from graphrag_pipeline.extraction.schemas import ExtractionRequest, Provenance
from graphrag_pipeline.ingestion.reconcile import ABSENT, COMPLETE, CONFLICT, PARTIAL, classify_chunk

SHA = "a" * 64


def make_request() -> ExtractionRequest:
    return ExtractionRequest(
        chunk_id="musique:r1:paragraph:0",
        record_id="r1",
        document_id="musique:document:example",
        paragraph_idx=0,
        text="The paragraph.",
        title="Example",
        provenance=Provenance(
            source_dataset="musique",
            source_split="fixture",
            source_file="fixture.jsonl",
            source_sha256=SHA,
            source_line=7,
        ),
    )


def document(**overrides) -> dict:
    base = {
        "chunk_id": "musique:r1:paragraph:0",
        "record_id": "r1",
        "document_id": "musique:document:example",
        "title": "Example",
        "source_dataset": "musique",
        "source_split": "fixture",
        "source_file": "fixture.jsonl",
        "source_sha256": SHA,
        "source_line": "7",
        "paragraph_idx": "0",
    }
    base.update(overrides)
    return base


def chunk(**overrides) -> dict:
    base = {
        "element_id": "c-1",
        "text": "The paragraph.",
        "chunk_id": "musique:r1:paragraph:0",
        "record_id": "r1",
        "document_id": "musique:document:example",
        "title": "Example",
        "source_file": "fixture.jsonl",
        "source_line": "7",
        "source_sha256": SHA,
        "paragraph_idx": 0,
    }
    base.update(overrides)
    return base


class FakeConnection:
    def __init__(self, rows):
        self.rows = rows

    def run_read(self, query, parameters=None):
        assert parameters == {"chunk_id": "musique:r1:paragraph:0"}
        return self.rows


def row(doc=None, chunks=None, element_id="d-1") -> dict:
    return {
        "document_element_id": element_id,
        "document": doc if doc is not None else document(),
        "chunks": [chunk()] if chunks is None else chunks,
    }


def classify(rows):
    return classify_chunk(FakeConnection(rows), make_request())


def test_absent_when_no_document() -> None:
    assert classify([]).state == ABSENT


def test_complete_when_document_and_single_chunk_match() -> None:
    assert classify([row()]).state == COMPLETE


def test_partial_when_document_has_no_chunk_and_report_locates_nodes() -> None:
    state = classify([row(chunks=[])])
    assert state.state == PARTIAL
    report = state.to_report(make_request())
    assert report["document_element_ids"] == ["d-1"]
    assert report["chunk_count"] == 0 and report["record_id"] == "r1"


def test_partial_when_paragraph_was_split_into_several_chunks() -> None:
    state = classify([row(chunks=[chunk(element_id="c-1"), chunk(element_id="c-2")])])
    assert state.state == PARTIAL
    assert state.chunk_element_ids == ["c-1", "c-2"]


def test_partial_when_chunk_lacks_copied_provenance() -> None:
    assert classify([row(chunks=[chunk(record_id=None)])]).state == PARTIAL


def test_conflict_when_chunk_text_differs() -> None:
    assert classify([row(chunks=[chunk(text="Different text.")])]).state == CONFLICT


def test_conflict_when_document_provenance_differs() -> None:
    assert classify([row(doc=document(source_sha256="b" * 64))]).state == CONFLICT
    assert classify([row(doc=document(record_id="other"))]).state == CONFLICT


def test_conflict_when_duplicate_documents_share_chunk_id() -> None:
    state = classify([row(element_id="d-1"), row(element_id="d-2")])
    assert state.state == CONFLICT
    assert state.document_element_ids == ["d-1", "d-2"]


@pytest.mark.parametrize("key", list(document()))
def test_any_document_provenance_mismatch_prevents_skip(key) -> None:
    assert classify([row(doc=document(**{key: "different"}))]).state == CONFLICT


@pytest.mark.parametrize("key", [key for key in chunk() if key not in ("element_id", "text")])
def test_any_missing_copied_chunk_provenance_prevents_skip(key) -> None:
    assert classify([row(chunks=[chunk(**{key: None})])]).state == PARTIAL


def test_optional_title_can_be_absent() -> None:
    request = make_request().model_copy(update={"title": None})
    connection = FakeConnection([row(doc=document(title=None), chunks=[chunk(title=None)])])
    assert classify_chunk(connection, request).state == COMPLETE
