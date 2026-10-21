"""Unit tests for the minimum Contract 2 extraction schemas.

Last modified: 2026-09-23

Purpose of this file:
    Verify the minimum Contract 2 request/result schemas, provenance
    preservation, mention references and
    exclusion of gold evaluation labels from the model-visible request.

    These tests exercise the boundary between a validated MuSiQue paragraph chunk
    and the extraction provider response. The fixture intentionally represents the
    model-visible request, so it excludes gold labels such as ``is_supporting``.

Files that use this file:
    Pytest discovers and runs this module. The tests import models from
    ``src/graphrag_pipeline/extraction/schemas.py`` and load
    ``tests/fixtures/musique/extraction_chunk.json``.

Caveats:
    This module does not implement or test the full Contract 1 record adapter,
    provider invocation, retry/dead-letter workflow, LangGraph orchestration,
    entity resolution, or Neo4j persistence. The tests validate schemas only.

    Contract 1's full record adapter, provider invocation, retry handling, and
    Neo4j persistence are outside this module. This module verifies only schema
    shape, provenance and identity preservation, and mention references.
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from graphrag_pipeline.extraction.schemas import (
    ExtractionRequest,
    ExtractionResult,
    validate_result_for_request,
)


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "musique"
FIXTURE_PATH = FIXTURE_DIR / "extraction_chunk.json"
SOURCE_RECORDS_PATH = FIXTURE_DIR / "valid_records.jsonl"


def load_request() -> ExtractionRequest:
    """Load and validate the focused paragraph request fixture.

    Inputs:
        None. The fixture path is defined by ``FIXTURE_PATH``.
    Outputs:
        A validated ``ExtractionRequest`` instance.
    """
    return ExtractionRequest.model_validate(json.loads(FIXTURE_PATH.read_text()))


def valid_result(request: ExtractionRequest) -> ExtractionResult:
    """Build a deterministic valid extraction result for the request text.

    Inputs:
        request: The validated Contract 2 extraction request whose text and
            provenance are copied into the result.
    Outputs:
        A validated ``ExtractionResult`` containing two mentions and one
        relationship.
    """
    text = request.text
    return ExtractionResult.model_validate(
        {
            "chunk_id": request.chunk_id,
            "record_id": request.record_id,
            "chunk_text": text,
            "provenance": request.provenance.model_dump(),
            "mentions": [
                {
                    "mention_id": "mention-1",
                    "mention": "Aster Systems",
                    "entity_type": "Organization",
                },
                {
                    "mention_id": "mention-2",
                    "mention": "Northwind Labs",
                    "entity_type": "Organization",
                },
            ],
            "triples": [
                {
                    "source_mention_id": "mention-1",
                    "relationship_type": "ACQUIRED",
                    "target_mention_id": "mention-2",
                }
            ],
        }
    )


def test_request_fixture_excludes_gold_support_label() -> None:
    """Ensure the Contract 2 model payload cannot expose evaluation labels.

    Inputs:
        None.
    Outputs:
        No return value; passes when the fixture request excludes
        ``is_supporting`` and contains the expected paragraph text.
    """
    request = load_request()

    assert request.text.startswith("Aster Systems acquired")
    assert "is_supporting" not in request.model_dump()


def test_extraction_fixture_preserves_source_paragraph() -> None:
    """Ensure the focused fixture is copied exactly from the ingestion fixture.

    Inputs:
        None. The focused extraction fixture declares its source file and line.
    Outputs:
        No return value; passes when source record identity, paragraph metadata,
        title, and text match exactly.
    """
    request = load_request()
    source_lines = SOURCE_RECORDS_PATH.read_text().splitlines()
    source_record = json.loads(source_lines[request.provenance.source_line - 1])
    source_paragraph = next(
        paragraph
        for paragraph in source_record["paragraphs"]
        if paragraph["idx"] == request.paragraph_idx
    )

    assert request.record_id == source_record["id"]
    assert request.title == source_paragraph["title"]
    assert request.text == source_paragraph["paragraph_text"]


def test_valid_result_preserves_contract_identity_and_provenance() -> None:
    """Ensure accepted output remains linked to its source chunk and run data.

    Inputs:
        None.
    Outputs:
        No return value; passes when result identity, provenance, and
        relationship endpoints are preserved.
    """
    request = load_request()

    result = validate_result_for_request(request, valid_result(request))

    assert result.chunk_id == request.chunk_id
    assert result.record_id == request.record_id
    assert result.provenance == request.provenance
    assert result.triples[0].relationship_type == "ACQUIRED"


def test_duplicate_mentions_are_rejected() -> None:
    """Reject duplicate mention IDs because references must be unambiguous.

    Inputs:
        None.
    Outputs:
        No return value; passes when duplicate mention IDs raise
        ``pydantic.ValidationError``.
    """
    request = load_request()
    result = valid_result(request).model_dump()
    result["mentions"][1]["mention_id"] = "mention-1"

    with pytest.raises(ValidationError, match="mention_id values must be unique"):
        ExtractionResult.model_validate(result)


def test_unknown_triple_mention_is_rejected() -> None:
    """Reject relationships that refer to mentions absent from the result.

    Inputs:
        None.
    Outputs:
        No return value; passes when an unknown target mention raises
        ``pydantic.ValidationError``.
    """
    request = load_request()
    result = valid_result(request).model_dump()
    result["triples"][0]["target_mention_id"] = "missing-mention"

    with pytest.raises(ValidationError, match="unknown target mention ID"):
        ExtractionResult.model_validate(result)


@pytest.mark.parametrize("field", ["confidence", "evidence_span"])
def test_retired_fields_are_not_part_of_new_output(field):
    result = valid_result(load_request()).model_dump()
    result["triples"][0][field] = 0.9
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ExtractionResult.model_validate(result)


def test_unknown_fields_are_rejected() -> None:
    """Reject fields outside the frozen Contract 2 schema.

    Inputs:
        None.
    Outputs:
        No return value; passes when an unexpected request field raises
        ``pydantic.ValidationError``.
    """
    payload = json.loads(FIXTURE_PATH.read_text())
    payload["is_supporting"] = True

    with pytest.raises(ValidationError, match="extra_forbidden"):
        ExtractionRequest.model_validate(payload)


def test_required_scalar_types_are_strict() -> None:
    """Reject string coercion for Contract 2 integer fields.

    Inputs:
        None.
    Outputs:
        No return value; passes when a string paragraph index raises
        ``pydantic.ValidationError`` instead of being coerced to an integer.
    """
    payload = json.loads(FIXTURE_PATH.read_text())
    payload["paragraph_idx"] = "1"

    with pytest.raises(ValidationError, match="int_type"):
        ExtractionRequest.model_validate(payload)