"""Versioned, atomic benchmark artifacts and strict offline validation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from graphrag_pipeline.extraction.schemas import ExtractionRequest, ExtractionResult, validate_result_for_request
from graphrag_pipeline.extraction_benchmark.contracts import ParagraphExtraction, validate_extraction_response
from graphrag_pipeline.extraction_benchmark.providers import SYSTEM_PROMPT, parse_model
from graphrag_pipeline.extraction_benchmark.scoring import SCORING_VERSION, score

FORMAT_VERSION = 1


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def reject_constant(value):
    raise ValueError(f"invalid JSON number {value}")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line, raw in enumerate(handle, 1):
            try:
                payload = json.loads(raw, parse_constant=reject_constant)
                if not isinstance(payload, dict):
                    raise ValueError("expected an object")
                rows.append(payload)
            except ValueError as error:
                raise ValueError(f"{path}:{line}: invalid artifact row: {error}") from error
    return rows


def atomic_text(path: Path, content: str):
    """Replace complete files; an interrupted write retains the previous snapshot."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path: Path, value: Any):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_jsonl(path: Path, rows):
    atomic_text(path, "".join(canonical(row) + "\n" for row in rows))


@contextmanager
def run_lock(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".benchmark.lock"
    try:
        with lock.open("x", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()}\n")
    except FileExistsError:
        raise ValueError(f"run is locked: {lock}; remove a stale lock only after confirming its process stopped") from None
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


class SavedExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    model: str
    chunk_id: str
    status: Literal["succeeded", "failed"]
    result: ExtractionResult | None
    raw_content: str | None
    error: dict | None
    attempts: StrictInt = Field(ge=1)
    timestamp: str
    elapsed_seconds: float = Field(ge=0)
    metadata: dict
    scores: dict | None = None

    @model_validator(mode="after")
    def validate_status(self):
        if self.status == "succeeded" and (self.result is None or self.raw_content is None or self.error is not None):
            raise ValueError("successful row needs a result/raw content and no error")
        if self.status == "failed" and (self.result is not None or not self.error):
            raise ValueError("failed row needs an error and no result")
        return self


def validate_rows(raw_rows, requests, model_specs, *, reference=None) -> dict[tuple[str, str], SavedExtraction]:
    by_id = {request.chunk_id: request for request in requests}
    rows = {}
    for raw in raw_rows:
        row = SavedExtraction.model_validate(raw)
        key = (row.model, row.chunk_id)
        if row.model not in model_specs or row.chunk_id not in by_id or key in rows:
            raise ValueError("unknown model/chunk or duplicate saved extraction")
        request = by_id[row.chunk_id]
        if row.result is not None:
            validate_result_for_request(request, row.result)
            parsed = validate_extraction_response(request, row.raw_content)
            if parsed != row.result:
                raise ValueError("saved result differs from validated raw content")
        expected_scores = score(reference[row.chunk_id], row.result) if reference is not None else None
        if canonical(row.scores) != canonical(expected_scores):
            raise ValueError("saved scores differ from deterministic comparison")
        rows[key] = row
    return rows


def load_run(directory: Path):
    manifest = read_json(directory / "manifest.json")
    if manifest.get("format_version") != FORMAT_VERSION or manifest.get("kind") not in {"gold", "candidate"}:
        raise ValueError("unsupported benchmark manifest")
    config = manifest["configuration"]
    if config.get("scoring_version") != SCORING_VERSION:
        raise ValueError("unsupported scoring version")
    requests = [ExtractionRequest.model_validate(raw) for raw in read_jsonl(directory / "cohort.jsonl")]
    cohort = [request.model_dump(mode="json") for request in requests]
    if not requests or len(requests) != config["paragraphs"] or len({r.chunk_id for r in requests}) != len(requests) or digest(cohort) != config["cohort_sha256"]:
        raise ValueError("cohort count/identity/checksum mismatch")
    specs = [f"{s['provider']}:{s['model']}" for s in config["models"]]
    if not specs or len(set(specs)) != len(specs):
        raise ValueError("missing or duplicate model configurations")
    for spec in specs:
        parse_model(spec)
    reference = None
    if manifest["kind"] == "candidate":
        raw_reference = read_jsonl(directory / "reference.jsonl")
        if digest(raw_reference) != config["reference_sha256"]:
            raise ValueError("frozen reference checksum mismatch")
        reference = {}
        for raw, request in zip(raw_reference, requests, strict=True):
            result = ExtractionResult.model_validate(raw)
            validate_result_for_request(request, result)
            reference[request.chunk_id] = result
    path = directory / ("gold.jsonl" if manifest["kind"] == "gold" else "predictions.jsonl")
    rows = validate_rows(read_jsonl(path) if path.exists() else [], requests, specs, reference=reference)
    return manifest, requests, rows, reference


def load_gold(directory: Path):
    manifest, requests, rows, _ = load_run(directory)
    if manifest["kind"] != "gold" or len(manifest["configuration"]["models"]) != 1:
        raise ValueError("--gold must identify a gold-generation directory")
    settings = manifest["configuration"]["models"][0]
    if manifest["configuration"].get("prompt") != SYSTEM_PROMPT or manifest["configuration"].get("response_schema") != ParagraphExtraction.model_json_schema():
        raise ValueError("gold prompt/schema differs from this benchmark version")
    if settings["provider"] != "openrouter" or settings.get("reasoning_effort") in {None, "none"}:
        raise ValueError("gold must use OpenRouter with reasoning enabled")
    if len(rows) != len(requests) or any(row.status != "succeeded" for row in rows.values()):
        raise ValueError("gold is incomplete or failed; resume gold generation before evaluating candidates")
    reference = {row.chunk_id: row.result for row in rows.values()}
    fingerprint = digest({"manifest": manifest, "rows": [row.model_dump(mode="json") for row in rows.values()]})
    return requests, reference, fingerprint
