"""Tests for the MuSiQue record-to-chunk adapter (Contract 1)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from graphrag_pipeline.ingestion.record_adapter import (
    RecordAdapterError,
    adapt_records_with_deduplication,
    build_context_links_for_record,
    build_requests_for_record,
    compute_file_sha256,
    compute_paragraph_hash,
    iter_jsonl_lines,
    normalize_text_whitespace,
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

    expected_hash_0 = compute_paragraph_hash("Aster Systems", "Aster Systems text.")
    expected_hash_1 = compute_paragraph_hash("Northwind Labs", "Northwind Labs text.")

    assert [request.chunk_id for request in requests] == [
        f"musique:paragraph:{expected_hash_0}",
        f"musique:paragraph:{expected_hash_1}",
    ]
    assert requests[0].paragraph_hash == expected_hash_0
    assert requests[1].paragraph_hash == expected_hash_1
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


# ==============================================================================
# Phase M3.1 Tests: Content-Hash Paragraph Identity
# ==============================================================================


def test_compute_paragraph_hash_synthetic_identical_and_variants() -> None:
    """Synthetic tests for M3.1:

    - Same text in two questions -> one hash
    - Whitespace variants (CRLF vs LF, margin whitespace) -> one hash
    - Changed title -> different hashes
    - Changed text -> different hashes
    """
    title = "Alan Turing"
    text = "Alan Turing was born in Maida Vale, London."

    # 1. Same text in two questions -> identical hash
    hash1 = compute_paragraph_hash(title, text)
    hash2 = compute_paragraph_hash(title, text)
    assert hash1 == hash2

    # 2. Whitespace variants -> identical hash
    crlf_text = "Alan Turing was born in Maida Vale, London.\r\n"
    padded_title = "  Alan Turing  "
    hash_variant = compute_paragraph_hash(padded_title, crlf_text)
    assert hash_variant == hash1

    # 3. Changed title -> different hash
    hash_diff_title = compute_paragraph_hash("Alan Mathison Turing", text)
    assert hash_diff_title != hash1

    # 4. Changed text -> different hash
    hash_diff_text = compute_paragraph_hash(title, "Alan Turing was born in London.")
    assert hash_diff_text != hash1


def test_compute_paragraph_hash_real_data_counts() -> None:
    """Real data tests for M3.1:

    Ans dev: 48,315 paragraph occurrences -> 21,100 unique hashes
    MVP slice: 9,996 paragraph occurrences -> 6,838 unique hashes
    """
    # Test on MVP slice gold jsonl
    slice_path = Path("eval/slices/ans_dev_mvp_v1_gold.jsonl")
    if slice_path.exists():
        with slice_path.open("r", encoding="utf-8") as f:
            slice_records = [json.loads(line) for line in f if line.strip()]

        slice_hashes = {
            compute_paragraph_hash(p["title"], p["paragraph_text"])
            for r in slice_records
            for p in r.get("paragraphs", [])
        }
        assert len(slice_hashes) == 6838

    # Test on full Ans dev if available
    ans_dev_path = Path("dataset/musique/data/data/musique_ans_v1.0_dev.jsonl")
    if ans_dev_path.exists():
        with ans_dev_path.open("r", encoding="utf-8") as f:
            ans_records = [json.loads(line) for line in f if line.strip()]

        ans_hashes = {
            compute_paragraph_hash(p["title"], p["paragraph_text"])
            for r in ans_records
            for p in r.get("paragraphs", [])
        }
        assert len(ans_hashes) == 21100


def test_build_context_links_for_record() -> None:
    """Context links map question key to paragraph index and content hash."""
    record = make_record("q_101", question="Who founded Aster Systems?")
    raw_line = json.dumps(record)

    links = build_context_links_for_record(raw_line, source_line=1)
    assert len(links) == 2

    assert links[0].question_key == "q_101"
    assert links[0].question_text == "Who founded Aster Systems?"
    assert links[0].paragraph_idx == 0
    assert links[0].paragraph_hash == compute_paragraph_hash("Aster Systems", "Aster Systems text.")

    assert links[1].question_key == "q_101"
    assert links[1].paragraph_idx == 1
    assert links[1].paragraph_hash == compute_paragraph_hash("Northwind Labs", "Northwind Labs text.")


def test_adapt_records_with_deduplication() -> None:
    """Shared paragraphs across records are deduplicated in requests while links are kept."""
    # Two questions sharing paragraph 0 ("Aster Systems")
    record_1 = make_record("q_1", question="Question 1")
    record_2 = {
        "id": "q_2",
        "question": "Question 2",
        "paragraphs": [
            {
                "idx": 0,
                "title": "Aster Systems",
                "paragraph_text": "Aster Systems text.",  # Shared with record 1
            },
            {
                "idx": 1,
                "title": "Third Company",
                "paragraph_text": "Third Company text.",
            },
        ],
    }

    lines = [
        (1, json.dumps(record_1)),
        (2, json.dumps(record_2)),
    ]

    requests, links, dead_letters, count = adapt_records_with_deduplication(
        iter(lines),
        source_file="fixture.jsonl",
        source_sha256="sha",
        source_split="dev",
    )

    assert count == 2
    assert not dead_letters
    # Total paragraphs = 4, but unique paragraphs = 3
    assert len(requests) == 3
    # All 4 context links preserved
    assert len(links) == 4
    # Unique requests contain Aster Systems only once
    aster_requests = [r for r in requests if r.title == "Aster Systems"]
    assert len(aster_requests) == 1
