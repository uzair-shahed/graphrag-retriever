"""Contract 1 adapter: raw MuSiQue JSONL records to validated Contract 2 chunks.

This module is the record-to-chunk boundary described in
``agent-coordination/data-contracts.md`` (Contract 1). It reads one official
MuSiQue JSONL line at a time, validates only the fields the paragraph-as-chunk
contract needs, and emits one ``ExtractionRequest`` per paragraph. Gold labels
such as ``is_supporting``, ``answer``, ``answer_aliases``, and
``question_decomposition`` are accepted on the raw record (so validation does
not fail because of them) but are never copied into the emitted request.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, ValidationError, model_validator

from graphrag_pipeline.extraction.schemas import ExtractionRequest, Provenance

_SLUG_INVALID_CHARS = re.compile(r"[^a-z0-9]+")


class RawMusiqueParagraph(BaseModel):
    """One paragraph inside an official MuSiQue JSONL record."""

    model_config = ConfigDict(extra="ignore")

    idx: StrictInt = Field(ge=0)
    title: StrictStr = Field(min_length=1)
    paragraph_text: StrictStr = Field(min_length=1)


class RawMusiqueRecord(BaseModel):
    """One official MuSiQue JSONL record validated for this project's needs."""

    model_config = ConfigDict(extra="ignore")

    id: StrictStr = Field(min_length=1)
    paragraphs: list[RawMusiqueParagraph] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_paragraph_indices(self) -> "RawMusiqueRecord":
        indices = [paragraph.idx for paragraph in self.paragraphs]
        if len(indices) != len(set(indices)):
            raise ValueError(f"record {self.id!r} has duplicate paragraph idx values")
        return self


class RecordAdapterError(ValueError):
    """Raised when one MuSiQue JSONL line cannot be adapted into chunks.

    Carries the source line and reason so a caller can record a dead-letter
    entry instead of silently skipping or rewriting the record.
    """

    def __init__(self, source_line: int, reason: str) -> None:
        super().__init__(f"line {source_line}: {reason}")
        self.source_line = source_line
        self.reason = reason


def slugify_title(title: str) -> str:
    """Normalize a paragraph title into a deterministic document ID segment."""
    slug = _SLUG_INVALID_CHARS.sub("-", title.strip().lower()).strip("-")
    if not slug:
        raise ValueError(f"title {title!r} does not normalize to a non-empty slug")
    return slug


def compute_file_sha256(path: Path) -> str:
    """Compute the SHA-256 checksum of an entire dataset file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_jsonl_lines(
    path: Path, start_record: int = 1, max_records: int | None = None
) -> Iterator[tuple[int, str]]:
    """Yield ``(source_line, raw_line)`` for a bounded slice of a JSONL file.

    ``start_record`` and ``max_records`` count only non-blank lines (one
    MuSiQue record per line), while the yielded ``source_line`` is the true
    1-indexed file line number, matching ``Provenance.source_line``.
    """
    if start_record < 1:
        raise ValueError("start_record must be 1 or greater")
    if max_records is not None and max_records < 1:
        raise ValueError("max_records must be 1 or greater when provided")

    record_index = 0
    yielded = 0
    with path.open("r", encoding="utf-8") as handle:
        for file_line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            record_index += 1
            if record_index < start_record:
                continue
            if max_records is not None and yielded >= max_records:
                return
            yielded += 1
            yield file_line_number, stripped


def build_requests_for_record(
    raw_line: str,
    source_line: int,
    source_file: str,
    source_sha256: str,
    source_split: str,
) -> list[ExtractionRequest]:
    """Adapt one raw MuSiQue JSONL line into one validated chunk per paragraph.

    Raises:
        RecordAdapterError: If the line is not valid JSON, does not satisfy the
            minimum MuSiQue record shape, contains duplicate paragraph indices,
            or contains a title that cannot be normalized into a document ID.
    """
    try:
        payload = json.loads(raw_line)
    except json.JSONDecodeError as error:
        raise RecordAdapterError(source_line, f"invalid JSON: {error}") from error

    try:
        record = RawMusiqueRecord.model_validate(payload)
    except ValidationError as error:
        raise RecordAdapterError(source_line, f"invalid MuSiQue record shape: {error}") from error

    provenance = Provenance(
        source_dataset="musique",
        source_split=source_split,
        source_file=source_file,
        source_sha256=source_sha256,
        source_line=source_line,
    )

    requests: list[ExtractionRequest] = []
    for paragraph in record.paragraphs:
        try:
            document_id = f"musique:document:{slugify_title(paragraph.title)}"
        except ValueError as error:
            raise RecordAdapterError(source_line, str(error)) from error
        requests.append(
            ExtractionRequest(
                chunk_id=f"musique:{record.id}:paragraph:{paragraph.idx}",
                record_id=record.id,
                document_id=document_id,
                paragraph_idx=paragraph.idx,
                text=paragraph.paragraph_text,
                title=paragraph.title,
                provenance=provenance,
            )
        )
    return requests
