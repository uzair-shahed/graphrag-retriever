"""Pydantic models for the minimum MuSiQue extraction contract.

Last modified: 2026-09-23

Purpose of this file:
    Define and validate the Contract 2 extraction request and result models,
    including provenance, mentions, and directed relationships.

Files that use this file:
    The extraction agent and its provider boundary use these models. The unit
    tests in ``tests/unit/test_extraction_schemas.py`` verify their behavior.

Caveats:
    The Contract 1 record adapter, provider invocation, retry/dead-letter
    workflow, LangGraph orchestration, entity resolution, and Neo4j persistence
    are not implemented here.

Contract coverage:

* Contract 1 (MuSiQue Record To Application Chunk) supplies the provenance
    envelope and chunk identity consumed by this module. The record adapter is
    responsible for creating those values; these models validate their shape.
* Contract 2 (Chunk To Extraction Agent) is implemented here for the minimum
    request and result schemas. ``ExtractionRequest`` is the model-visible
    payload, while ``ExtractionResult`` is the validated provider response.
* Contract 2's provider calls, retry/dead-letter workflow, and execution trace
    remain outside this module and belong to the extraction agent boundary.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator


class Provenance(BaseModel):
    """Contract 1 provenance required to carry a chunk into Contract 2."""

    model_config = ConfigDict(extra="forbid")

    source_dataset: StrictStr = Field(min_length=1)
    source_split: StrictStr = Field(min_length=1)
    source_file: StrictStr = Field(min_length=1)
    source_sha256: StrictStr = Field(min_length=1)
    source_line: StrictInt = Field(ge=1)


class ExtractionRequest(BaseModel):
    """Contract 2 input: one validated chunk without gold labels."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: StrictStr = Field(min_length=1)
    record_id: StrictStr = Field(min_length=1)
    document_id: StrictStr = Field(min_length=1)
    paragraph_idx: StrictInt = Field(ge=0)
    text: StrictStr = Field(min_length=1)
    title: StrictStr | None = None
    provenance: Provenance


class Mention(BaseModel):
    """Extracted entity mention with a paragraph-local relationship identifier."""

    model_config = ConfigDict(extra="forbid")

    mention_id: StrictStr = Field(min_length=1)
    mention: StrictStr = Field(min_length=1)
    entity_type: StrictStr = Field(min_length=1)


class Triple(BaseModel):
    """Directed relationship between two paragraph-local mentions."""

    model_config = ConfigDict(extra="forbid")

    source_mention_id: StrictStr = Field(min_length=1)
    relationship_type: StrictStr = Field(min_length=1)
    target_mention_id: StrictStr = Field(min_length=1)


class ExtractionResult(BaseModel):
    """Contract 2 validated output containing mentions and relationships."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: StrictStr = Field(min_length=1)
    record_id: StrictStr = Field(min_length=1)
    chunk_text: StrictStr = Field(min_length=1)
    provenance: Provenance
    mentions: list[Mention]
    triples: list[Triple]

    @model_validator(mode="after")
    def validate_internal_references(self) -> Self:
        mention_ids = [mention.mention_id for mention in self.mentions]
        if len(mention_ids) != len(set(mention_ids)):
            raise ValueError("mention_id values must be unique")

        mention_id_set = set(mention_ids)
        for triple in self.triples:
            if triple.source_mention_id not in mention_id_set:
                raise ValueError(
                    f"unknown source mention ID: {triple.source_mention_id}"
                )
            if triple.target_mention_id not in mention_id_set:
                raise ValueError(
                    f"unknown target mention ID: {triple.target_mention_id}"
                )
        return self


def validate_result_for_request(
    request: ExtractionRequest, result: ExtractionResult
) -> ExtractionResult:
    """Apply Contract 2 request/result identity and provenance checks."""
    if result.chunk_id != request.chunk_id:
        raise ValueError("extraction result chunk_id does not match the request")
    if result.record_id != request.record_id:
        raise ValueError("extraction result record_id does not match the request")
    if result.chunk_text != request.text:
        raise ValueError("extraction result chunk_text does not match the request text")
    if result.provenance != request.provenance:
        raise ValueError("extraction result provenance does not match the request")

    return result
