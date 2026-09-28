"""Tests for the MuSiQue record-to-chunk adapter (Contract 1)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from graphrag_pipeline.ingestion.record_adapter import (
    RecordAdapterError,
    build_requests_for_record,
    compute_file_sha256,
    iter_jsonl_lines,
    slugify_title,
)


def make_record(record_id: str = "2hop__1_2", **overrides: object) -> dict:
    """Build a raw MuSiQue-shaped record dict, including gold-label fields."""
    record = {
        "id": record_id,
        "question": "gold question text",
        "answer": "gold answer",
        "answer_aliases": ["gold alias"],
        "answerable": True,
        "question_decomposition": [
            {"id": 1, "question": "q", "answer": "a", "paragraph_support_idx": 0}
        ],
        "paragraphs": [
            {
                "idx": 0,
                "title": "Aster Systems",
                "paragraph_text": "Aster Systems text.",
                "is_supporting": True,
            },
            {
                "idx": 1,
                "title": "Northwind Labs",
                "paragraph_text": "Northwind Labs text.",
                "is_supporting": False,
            },
        ],
    }
    record.update(overrides)
    return record


def test_build_requests_for_record_creates_one_chunk_per_paragraph() -> None:
    """Each paragraph becomes its own chunk with deterministic IDs and provenance."""
    raw_line = json.dumps(make_record())

    requests = build_requests_for_record(
        raw_line=raw_line,
        source_line=7,
        source_file="dataset/musique/data/data/musique_full_v1.0_dev.jsonl",
        source_sha256="abc123",
        source_split="full_dev",
    )

    assert [request.chunk_id for request in requests] == [
        "musique:2hop__1_2:paragraph:0",
        "musique:2hop__1_2:paragraph:1",
    ]
    assert requests[0].document_id == "musique:document:aster-systems"
    assert requests[0].provenance.source_line == 7
    assert requests[0].provenance.source_split == "full_dev"
    assert requests[0].text == "Aster Systems text."


def test_build_requests_for_record_excludes_gold_fields() -> None:
    """Gold labels present on the raw record never reach the emitted request."""
    raw_line = json.dumps(make_record())

    requests = build_requests_for_record(
        raw_line=raw_line,
        source_line=1,
        source_file="fixture.jsonl",
        source_sha256="abc123",
        source_split="fixture",
    )

    payload = requests[0].model_dump()
    assert "is_supporting" not in payload
    assert "answer" not in payload
    assert "answer_aliases" not in payload
    assert "question_decomposition" not in payload


def test_build_requests_for_record_rejects_duplicate_paragraph_idx() -> None:
    """Duplicate paragraph idx values dead-letter the whole record."""
    record = make_record()
    record["paragraphs"][1]["idx"] = 0
    raw_line = json.dumps(record)

    with pytest.raises(RecordAdapterError):
        build_requests_for_record(
            raw_line=raw_line,
            source_line=1,
            source_file="fixture.jsonl",
            source_sha256="abc123",
            source_split="fixture",
        )


def test_build_requests_for_record_rejects_missing_paragraphs_field() -> None:
    """A record missing the required paragraphs field is rejected."""
    record = make_record()
    del record["paragraphs"]
    raw_line = json.dumps(record)

    with pytest.raises(RecordAdapterError):
        build_requests_for_record(
            raw_line=raw_line,
            source_line=1,
            source_file="fixture.jsonl",
            source_sha256="abc123",
            source_split="fixture",
        )


def test_build_requests_for_record_rejects_malformed_json() -> None:
    """A line that is not valid JSON raises RecordAdapterError."""
    with pytest.raises(RecordAdapterError):
        build_requests_for_record(
            raw_line="{not valid json",
            source_line=1,
            source_file="fixture.jsonl",
            source_sha256="abc123",
            source_split="fixture",
        )


def test_slugify_title_normalizes_punctuation_and_case() -> None:
    """Titles are lower-cased and punctuation collapses to single hyphens."""
    assert slugify_title("Aster Systems, Inc.") == "aster-systems-inc"


def test_slugify_title_rejects_empty_result() -> None:
    """A title with no alphanumeric characters cannot form a slug."""
    with pytest.raises(ValueError):
        slugify_title("***")


def test_compute_file_sha256_matches_hashlib(tmp_path: Path) -> None:
    """The streamed checksum matches a direct hashlib digest of the same file."""
    file_path = tmp_path / "sample.jsonl"
    file_path.write_text('{"id": "1"}\n', encoding="utf-8")

    expected = hashlib.sha256(file_path.read_bytes()).hexdigest()
    assert compute_file_sha256(file_path) == expected


def test_iter_jsonl_lines_applies_start_and_limit(tmp_path: Path) -> None:
    """start_record and max_records select the correct record window and line numbers."""
    file_path = tmp_path / "sample.jsonl"
    file_path.write_text(
        "\n".join(json.dumps({"id": str(i)}) for i in range(1, 6)) + "\n", encoding="utf-8"
    )

    results = list(iter_jsonl_lines(file_path, start_record=2, max_records=2))

    assert [line_number for line_number, _ in results] == [2, 3]
    assert json.loads(results[0][1])["id"] == "2"
    assert json.loads(results[1][1])["id"] == "3"


def test_iter_jsonl_lines_skips_blank_lines(tmp_path: Path) -> None:
    """Blank lines do not count toward start_record or max_records."""
    file_path = tmp_path / "sample.jsonl"
    file_path.write_text('{"id": "1"}\n\n{"id": "2"}\n', encoding="utf-8")

    results = list(iter_jsonl_lines(file_path, start_record=1, max_records=2))

    assert [json.loads(raw)["id"] for _, raw in results] == ["1", "2"]
