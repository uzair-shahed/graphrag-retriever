"""Pydantic models for the minimum MuSiQue extraction contract.

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

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Provenance(BaseModel):
    """Contract 1 provenance required to carry a chunk into Contract 2."""

    model_config = ConfigDict(extra="forbid")

    source_dataset: str = Field(min_length=1)
    source_split: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    source_sha256: str = Field(min_length=1)
    source_line: int = Field(ge=1)


class ExtractionRequest(BaseModel):
    """Contract 2 input: one validated chunk without gold labels."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    record_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    paragraph_idx: int = Field(ge=0)
    text: str = Field(min_length=1)
    title: str | None = None
    provenance: Provenance


class EvidenceSpan(BaseModel):
    """Contract 2 evidence span with offsets into the original chunk text."""

    model_config = ConfigDict(extra="forbid")

    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_bounds_and_text(self) -> Self:
        if self.end <= self.start:
            raise ValueError("evidence span end must be greater than start")
        return self


class Mention(BaseModel):
    """Contract 2 extracted entity mention and its source span."""

    model_config = ConfigDict(extra="forbid")

    mention_id: str = Field(min_length=1)
    mention: str = Field(min_length=1)
    entity_type: str = Field(min_length=1)
    evidence_span: EvidenceSpan


class Triple(BaseModel):
    """Contract 2 extracted relationship and confidence score."""

    model_config = ConfigDict(extra="forbid")

    source_mention_id: str = Field(min_length=1)
    relationship_type: str = Field(min_length=1)
    target_mention_id: str = Field(min_length=1)
    evidence_span: EvidenceSpan
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractionResult(BaseModel):
    """Contract 2 validated output containing mentions and relationships."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    record_id: str = Field(min_length=1)
    chunk_text: str = Field(min_length=1)
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

    for collection in (result.mentions, result.triples):
        for item in collection:
            span = item.evidence_span
            if span.end > len(request.text):
                raise ValueError("evidence span exceeds request text length")
            if request.text[span.start : span.end] != span.text:
                raise ValueError("evidence span text does not match request text")

    return result