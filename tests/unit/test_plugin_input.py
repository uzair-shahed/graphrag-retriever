"""Tests for the MuSiQue-to-SimpleKGPipeline input contract."""

import json
from pathlib import Path

from graphrag_pipeline.extraction.plugin_input import build_plugin_input
from graphrag_pipeline.extraction.schemas import ExtractionRequest


FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "musique" / "extraction_chunk.json"


def load_request() -> ExtractionRequest:
    return ExtractionRequest.model_validate(json.loads(FIXTURE_PATH.read_text()))


def test_plugin_input_preserves_text_and_provenance() -> None:
    request = load_request()

    text, metadata = build_plugin_input(request)

    assert text == request.text
    assert metadata == {
        "source_dataset": "musique",
        "source_split": "fixture",
        "source_file": "tests/fixtures/musique/valid_records.jsonl",
        "source_sha256": "fixture-sha256",
        "source_line": 1,
        "record_id": "fixture_valid_multi_hop_001",
        "chunk_id": "musique:fixture_valid_multi_hop_001:paragraph:1",
        "document_id": "musique:document:aster-systems-acquisition",
        "paragraph_idx": 1,
        "title": "Aster Systems Acquisition",
    }


def test_plugin_input_excludes_gold_and_model_irrelevant_fields() -> None:
    request = load_request()

    text, metadata = build_plugin_input(request)
    payload = {"text": text, "document_metadata": metadata}

    assert "is_supporting" not in payload
    assert "answer" not in payload
    assert "answer_aliases" not in payload
    assert "question_decomposition" not in payload
