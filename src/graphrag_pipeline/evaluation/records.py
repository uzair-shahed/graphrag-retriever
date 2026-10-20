"""Validated gold records and selection shared by preparation and evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator

from graphrag_pipeline.ingestion.record_adapter import (
    RawMusiqueParagraph, build_requests_for_record, compute_file_sha256, iter_jsonl_lines,
)
from graphrag_pipeline.query import QueryRequest

DEFAULT_DATASET = Path("dataset/musique/data/data/musique_ans_v1.0_dev.jsonl")


class GoldParagraph(RawMusiqueParagraph):
    is_supporting: StrictBool


class GoldHop(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: StrictInt
    question: StrictStr = Field(min_length=1)
    answer: StrictStr | None
    paragraph_support_idx: StrictInt | None = Field(ge=0)


class GoldRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: StrictStr = Field(min_length=1)
    question: StrictStr = Field(min_length=1)
    answer: StrictStr | None
    answer_aliases: list[StrictStr] = Field(default_factory=list)
    answerable: StrictBool
    paragraphs: list[GoldParagraph] = Field(min_length=1)
    question_decomposition: list[GoldHop] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_gold(self):
        indices = [p.idx for p in self.paragraphs]
        if len(indices) != len(set(indices)):
            raise ValueError("duplicate paragraph indices")
        if len({h.id for h in self.question_decomposition}) != len(self.question_decomposition):
            raise ValueError("duplicate decomposition hop IDs")
        for hop in self.question_decomposition:
            if hop.paragraph_support_idx is not None and hop.paragraph_support_idx not in indices:
                raise ValueError("decomposition support refers to a missing paragraph")
        if self.answerable:
            if not self.answer or not self.answer.strip():
                raise ValueError("answerable record requires a gold answer")
            if any(h.paragraph_support_idx is None for h in self.question_decomposition):
                raise ValueError("answerable record requires support for every hop")
            if {h.paragraph_support_idx for h in self.question_decomposition} != {p.idx for p in self.paragraphs if p.is_supporting}:
                raise ValueError("supporting paragraphs and decomposition support disagree")
        return self


@dataclass(frozen=True)
class EvaluationCase:
    gold: GoldRecord
    source_file: str
    source_line: int
    source_sha256: str

    def query(self, trace_id: str, backend: str = "vectorrag") -> QueryRequest:
        """Only non-label fields may cross the retrieval/answer boundary."""
        return QueryRequest(
            question=self.gold.question, trace_id=trace_id, origin="musique",
            record_id=self.gold.id, source_split="ans_dev", source_file=self.source_file,
            source_line=self.source_line, paragraph_indices=[p.idx for p in self.gold.paragraphs],
            paragraph_titles=[p.title for p in self.gold.paragraphs], retrieval_backend=backend,
        )

    def extraction_requests(self):
        return build_requests_for_record(
            self.gold.model_dump_json(), self.source_line, self.source_file, self.source_sha256, "ans_dev",
        )


def load_cases(
    path: Path = DEFAULT_DATASET, *, record_ids: list[str] | None = None,
    start_record: int = 1, records: int | None = None,
) -> list[EvaluationCase]:
    """Select exact IDs or nonblank record positions, matching the batch loader.

    Reaching EOF returns the available range, just as ingestion does. Empty
    selections, unknown IDs, and ambiguous mixed selectors are errors.
    """
    if path.name != DEFAULT_DATASET.name:
        raise ValueError(f"evaluation currently supports only {DEFAULT_DATASET.name}")
    if start_record < 1 or (records is not None and records < 1):
        raise ValueError("start-record and records must be positive")
    if record_ids and (start_record != 1 or records is not None):
        raise ValueError("record IDs cannot be combined with range selection")
    if record_ids and len(record_ids) != len(set(record_ids)):
        raise ValueError("duplicate selected record IDs")
    checksum = compute_file_sha256(path)
    cases = {}
    seen = set()
    for position, (source_line, raw) in enumerate(iter_jsonl_lines(path), 1):
        payload = json.loads(raw)
        record_id = payload.get("id")
        if not isinstance(record_id, str) or record_id in seen:
            raise ValueError(f"{path}:{source_line}: missing or duplicate record ID")
        seen.add(record_id)
        selected = record_id in record_ids if record_ids else position >= start_record and (records is None or position < start_record + records)
        if selected:
            cases[record_id] = EvaluationCase(GoldRecord.model_validate(payload), str(path), source_line, checksum)
    if record_ids:
        missing = set(record_ids) - cases.keys()
        if missing:
            raise ValueError(f"unknown selected record IDs: {sorted(missing)}")
        result = [cases[r] for r in record_ids]
    else:
        result = list(cases.values())
    if not result:
        raise ValueError("selection contains no records")
    return result
