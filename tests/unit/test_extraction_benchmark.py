"""Paragraph selection and schema boundaries, without model or database calls."""

import hashlib
import json

import pytest

from graphrag_pipeline.extraction.schemas import ExtractionResult
from graphrag_pipeline.extraction_benchmark.contracts import (
    ParagraphExtraction,
    model_input,
    validate_extraction_response,
)
from graphrag_pipeline.extraction_benchmark.records import load_paragraphs


def record(record_id, indices):
    return {
        "id": record_id,
        "question": "QUESTION_SECRET",
        "answer": "ANSWER_SECRET",
        "answer_aliases": ["ALIAS_SECRET"],
        "answerable": True,
        "question_decomposition": [{"question": "HOP_SECRET"}],
        "paragraphs": [
            {"idx": idx, "title": f"Title {idx}",
             "paragraph_text": "  Alice met Bob.\n", "is_supporting": True}
            for idx in indices
        ],
    }


def write_records(tmp_path, records, *, leading_blank=False):
    path = tmp_path / "musique_ans_v1.0_dev.jsonl"
    path.write_text(("\n" if leading_blank else "") + "\n".join(
        json.dumps(r) for r in records
    ) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def paragraph(tmp_path):
    path = write_records(tmp_path, [record("source-record", [3])])
    return load_paragraphs(path, count=1)[0]


def extraction_payload(paragraph):
    text = paragraph.text
    return {
        "mentions": [
            {"mention_id": name.lower(), "mention": name, "entity_type": "Person",
             "evidence_span": {"start": text.index(name),
                               "end": text.index(name) + len(name), "text": name}}
            for name in ("Alice", "Bob")
        ],
        "triples": [{
            "source_mention_id": "alice", "relationship_type": "MET",
            "target_mention_id": "bob", "confidence": 0.99,
            "evidence_span": {"start": 0, "end": len(text), "text": text},
        }],
    }


def test_default_cohort_is_150_paragraphs_in_source_list_order(tmp_path):
    # Unsorted indices prove selection follows the list rather than idx sorting.
    path = write_records(tmp_path, [
        record("first", [9, 2, 7]), record("second", range(200)),
    ], leading_blank=True)

    paragraphs = load_paragraphs(path)

    assert len(paragraphs) == 150
    assert [p.paragraph_idx for p in paragraphs[:3]] == [9, 2, 7]
    assert paragraphs[3].chunk_id == "musique:second:paragraph:0"
    assert paragraphs[-1].chunk_id == "musique:second:paragraph:146"
    assert paragraphs[0].provenance.source_line == 2
    assert paragraphs[3].provenance.source_line == 3
    assert paragraphs[0].provenance.source_file == str(path)
    assert paragraphs[0].provenance.source_split == "ans_dev"
    assert paragraphs[0].provenance.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len({p.chunk_id for p in paragraphs}) == 150
    assert all(p.text == "  Alice met Bob.\n" for p in paragraphs)
    assert paragraphs == load_paragraphs(path)


def test_model_input_contains_only_title_and_original_text(paragraph):
    assert model_input(paragraph) == {"title": "Title 3", "text": "  Alice met Bob.\n"}
    dumped = paragraph.model_dump_json()
    for secret in ("QUESTION_SECRET", "ANSWER_SECRET", "ALIAS_SECRET", "HOP_SECRET"):
        assert secret not in dumped
    assert "is_supporting" not in dumped
    assert set(model_input(paragraph)) == {"title", "text"}


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_count_fails_before_opening_dataset(tmp_path, count):
    with pytest.raises(ValueError, match="positive integer"):
        load_paragraphs(tmp_path / "missing.jsonl", count=count)


@pytest.mark.parametrize("records, expected", [([], 0), ([record("one", [0])], 1)])
def test_short_dataset_cannot_silently_shrink_cohort(tmp_path, records, expected):
    path = write_records(tmp_path, records)
    with pytest.raises(ValueError, match=f"contains only {expected}"):
        load_paragraphs(path, count=2)


def test_duplicate_record_ids_are_rejected(tmp_path):
    path = write_records(tmp_path, [record("duplicate", [0]), record("duplicate", [1])])
    with pytest.raises(ValueError, match="duplicate record ID"):
        load_paragraphs(path, count=2)


def test_duplicate_paragraph_indices_are_rejected(tmp_path):
    path = write_records(tmp_path, [record("one", [0, 0])])
    with pytest.raises(ValueError, match="duplicate paragraph idx"):
        load_paragraphs(path, count=2)


def test_malformed_selected_record_is_not_skipped(tmp_path):
    path = write_records(tmp_path, [record("one", [0])])
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{invalid json}\n")
    with pytest.raises(ValueError, match="line 2: invalid JSON"):
        load_paragraphs(path, count=2)
    assert len(load_paragraphs(path, count=1)) == 1


def test_output_schema_reuses_existing_mentions_and_triples():
    schema = ParagraphExtraction.model_json_schema()
    shared = ExtractionResult.model_json_schema()
    assert schema["required"] == ["mentions", "triples"]
    assert schema["additionalProperties"] is False
    for name in ("Mention", "Triple", "EvidenceSpan"):
        assert schema["$defs"][name] == shared["$defs"][name]
    assert schema["properties"]["mentions"]["items"] == {"$ref": "#/$defs/Mention"}
    assert schema["properties"]["triples"]["items"] == {"$ref": "#/$defs/Triple"}
    assert "Provenance" not in schema["$defs"]


def test_valid_output_binds_local_identity_and_validates_shared_contract(paragraph):
    result = validate_extraction_response(paragraph, json.dumps(extraction_payload(paragraph)))
    assert isinstance(result, ExtractionResult)
    assert result.chunk_id == paragraph.chunk_id
    assert result.record_id == paragraph.record_id
    assert result.chunk_text == paragraph.text
    assert result.provenance == paragraph.provenance
    assert result.triples[0].source_mention_id == "alice"


def test_empty_extraction_is_valid(paragraph):
    result = validate_extraction_response(paragraph, '{"mentions": [], "triples": []}')
    assert result.mentions == [] and result.triples == []


@pytest.mark.parametrize("mutation, message", [
    ("unknown_endpoint", "unknown target mention ID"),
    ("duplicate_id", "mention_id values must be unique"),
    ("span_text", "evidence span text does not match"),
    ("span_bounds", "evidence span exceeds"),
    ("relation_span", "evidence span text does not match"),
    ("confidence", "less than or equal to 1"),
    ("extra_identity", "Extra inputs are not permitted"),
    ("wrong_type", "valid integer"),
])
def test_invalid_output_cannot_become_gold_or_prediction(paragraph, mutation, message):
    payload = extraction_payload(paragraph)
    if mutation == "unknown_endpoint":
        payload["triples"][0]["target_mention_id"] = "missing"
    elif mutation == "duplicate_id":
        payload["mentions"][1]["mention_id"] = "alice"
    elif mutation == "span_text":
        payload["mentions"][0]["evidence_span"]["text"] = "wrong"
    elif mutation == "span_bounds":
        payload["mentions"][0]["evidence_span"]["end"] = len(paragraph.text) + 1
    elif mutation == "relation_span":
        payload["triples"][0]["evidence_span"]["text"] = "wrong"
    elif mutation == "confidence":
        payload["triples"][0]["confidence"] = 1.1
    elif mutation == "extra_identity":
        payload["record_id"] = "injected"
    elif mutation == "wrong_type":
        payload["mentions"][0]["evidence_span"]["start"] = "2"
    with pytest.raises(ValueError, match=message):
        validate_extraction_response(paragraph, json.dumps(payload))


@pytest.mark.parametrize("content", ["", "not JSON", '{"mentions": []}', "null"])
def test_malformed_or_incomplete_output_is_rejected(paragraph, content):
    with pytest.raises(ValueError):
        validate_extraction_response(paragraph, content)
