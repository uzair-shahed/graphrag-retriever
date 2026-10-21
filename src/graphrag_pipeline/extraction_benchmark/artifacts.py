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

FORMAT_VERSION = 2
APPEND_TRANSACTION = ".append-transaction.json"


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
    response_format_version: StrictInt = Field(default=2, ge=1, le=2)

    @model_validator(mode="after")
    def validate_status(self):
        if self.status == "succeeded" and (self.result is None or self.raw_content is None or self.error is not None):
            raise ValueError("successful row needs a result/raw content and no error")
        if self.status == "failed" and (self.result is not None or not self.error):
            raise ValueError("failed row needs an error and no result")
        return self


class SkippedGold(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    reason: Literal["gold_failed", "gold_missing"]
    error_type: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_reason(self):
        if (self.reason == "gold_failed") != (self.error_type is not None):
            raise ValueError("failed gold needs an error type; missing gold has no error type")
        return self


def validate_gold_coverage(requests, reference, skipped_gold):
    """Every requested paragraph has exactly one usable label or skip record."""
    ids = {request.chunk_id for request in requests}
    skips = [SkippedGold.model_validate(raw) for raw in skipped_gold]
    by_id = {skip.chunk_id: skip for skip in skips}
    if (len(by_id) != len(skips) or set(by_id) & set(reference)
            or set(by_id) | set(reference) != ids):
        raise ValueError("gold reference/skip coverage mismatch or duplicate skipped paragraph")
    for request in requests:
        if request.chunk_id in reference:
            validate_result_for_request(request, reference[request.chunk_id])
    return [by_id[r.chunk_id].model_dump(mode="json") for r in requests if r.chunk_id in by_id]


def validate_rows(raw_rows, requests, model_specs, *, reference=None,
                  legacy_rows=None) -> dict[tuple[str, str], SavedExtraction]:
    by_id = {request.chunk_id: request for request in requests}
    rows = {}
    for raw in raw_rows:
        row = SavedExtraction.model_validate(raw)
        key = (row.model, row.chunk_id)
        if row.model not in model_specs or row.chunk_id not in by_id or key in rows:
            raise ValueError("unknown model/chunk or duplicate saved extraction")
        if reference is not None and row.chunk_id not in reference:
            raise ValueError("saved prediction has no usable gold label")
        if row.response_format_version == 1:
            original = (legacy_rows or {}).get(key)
            if original is None or original.get("raw_content_sha256") != digest(row.raw_content):
                raise ValueError("legacy response is missing migration provenance or its raw content changed")
        request = by_id[row.chunk_id]
        if row.result is not None:
            validate_result_for_request(request, row.result)
            parsed = validate_extraction_response(request, row.raw_content,
                                                  response_format_version=row.response_format_version)
            if parsed != row.result:
                raise ValueError("saved result differs from validated raw content")
        expected_scores = score(reference[row.chunk_id], row.result) if reference is not None else None
        if canonical(row.scores) != canonical(expected_scores):
            raise ValueError("saved scores differ from deterministic comparison")
        rows[key] = row
    return rows


def load_run(directory: Path):
    manifest = read_json(directory / "manifest.json")
    if manifest.get("format_version") == 1:
        raise ValueError("benchmark format v1 requires offline conversion: use migrate --input OLD_RUN --output NEW_RUN (and --gold CONVERTED_GOLD for candidates)")
    transaction_path = directory / APPEND_TRANSACTION
    transaction = read_json(transaction_path) if transaction_path.exists() else None
    if transaction is not None:
        target = transaction["manifest"]
        if (transaction.get("format_version") != FORMAT_VERSION
                or digest(manifest) not in {transaction["previous_manifest_sha256"], digest(target)}
                or target.get("run_id") != manifest.get("run_id")
                or target.get("kind") != manifest.get("kind")):
            raise ValueError("invalid append transaction; saved manifest differs")
        # Read the committed target snapshot even if only some files were replaced.
        # Reading (including --dry-run) never repairs or writes artifacts.
        manifest = target
    if manifest.get("format_version") != FORMAT_VERSION or manifest.get("kind") not in {"gold", "candidate"}:
        raise ValueError("unsupported benchmark manifest")
    config = manifest["configuration"]
    if config.get("scoring_version") != SCORING_VERSION:
        raise ValueError("unsupported scoring version")
    raw_cohort = transaction["cohort"] if transaction is not None else read_jsonl(directory / "cohort.jsonl")
    requests = [ExtractionRequest.model_validate(raw) for raw in raw_cohort]
    cohort = [request.model_dump(mode="json") for request in requests]
    if not requests or len(requests) != config["paragraphs"] or len({r.chunk_id for r in requests}) != len(requests) or digest(cohort) != config["cohort_sha256"]:
        raise ValueError("cohort count/identity/checksum mismatch")
    specs = [f"{s['provider']}:{s['model']}" for s in config["models"]]
    if not specs or len(set(specs)) != len(specs):
        raise ValueError("missing or duplicate model configurations")
    for spec in specs:
        parse_model(spec)
    legacy_rows = {}
    if "migration" in manifest:
        audit = read_json(directory / "migration.json")
        if digest(audit) != manifest["migration"]["audit_sha256"]:
            raise ValueError("migration audit checksum mismatch")
        for row in audit["rows"]:
            key = (row["model"], row["chunk_id"])
            if key in legacy_rows:
                raise ValueError("duplicate migration audit row")
            legacy_rows[key] = row
    reference = None
    if manifest["kind"] == "candidate":
        raw_reference = transaction["reference"] if transaction is not None else read_jsonl(directory / "reference.jsonl")
        if digest(raw_reference) != config["reference_sha256"]:
            raise ValueError("frozen reference checksum mismatch")
        reference = {}
        skipped_gold = config.get("skipped_gold", [])
        skipped_ids = {SkippedGold.model_validate(raw).chunk_id for raw in skipped_gold}
        eligible = [request for request in requests if request.chunk_id not in skipped_ids]
        for raw, request in zip(raw_reference, eligible, strict=True):
            result = ExtractionResult.model_validate(raw)
            validate_result_for_request(request, result)
            reference[request.chunk_id] = result
        if validate_gold_coverage(requests, reference, skipped_gold) != skipped_gold:
            raise ValueError("skipped gold must follow cohort order")
    path = directory / ("gold.jsonl" if manifest["kind"] == "gold" else "predictions.jsonl")
    rows = validate_rows(read_jsonl(path) if path.exists() else [], requests, specs, reference=reference,
                         legacy_rows=legacy_rows)
    return manifest, requests, rows, reference


def recover_append(directory: Path):
    """Finish an interrupted metadata commit. Caller must hold the run lock."""
    journal = directory / APPEND_TRANSACTION
    if not journal.exists():
        return
    manifest, requests, _, reference = load_run(directory)
    write_jsonl(directory / "cohort.jsonl", [request.model_dump(mode="json") for request in requests])
    if reference is not None:
        write_jsonl(directory / "reference.jsonl", [reference[request.chunk_id].model_dump(mode="json")
                                                  for request in requests if request.chunk_id in reference])
    write_json(directory / "manifest.json", manifest)
    journal.unlink()


def commit_append(directory: Path, previous_manifest, manifest, cohort, reference):
    """Journal the full target before replacing metadata; never rewrite results."""
    write_json(directory / APPEND_TRANSACTION, {
        "format_version": FORMAT_VERSION,
        "previous_manifest_sha256": digest(previous_manifest),
        "manifest": manifest, "cohort": cohort, "reference": reference,
    })
    recover_append(directory)


def load_gold(directory: Path):
    """Return full cohort, usable references, fingerprint and ordered skip records."""
    manifest, requests, rows, _ = load_run(directory)
    if manifest["kind"] != "gold" or len(manifest["configuration"]["models"]) != 1:
        raise ValueError("--gold must identify a gold-generation directory")
    settings = manifest["configuration"]["models"][0]
    if manifest["configuration"].get("prompt") != SYSTEM_PROMPT or manifest["configuration"].get("response_schema") != ParagraphExtraction.model_json_schema():
        raise ValueError("gold prompt/schema differs from this benchmark version")
    if settings["provider"] != "openrouter" or settings.get("reasoning_effort") in {None, "none"}:
        raise ValueError("gold must use OpenRouter with reasoning enabled")
    reference = {row.chunk_id: row.result for row in rows.values() if row.status == "succeeded"}
    by_id = {row.chunk_id: row for row in rows.values()}
    skipped_gold = []
    for request in requests:
        row = by_id.get(request.chunk_id)
        if row is None:
            skipped_gold.append({"chunk_id": request.chunk_id, "reason": "gold_missing", "error_type": None})
        elif row.status == "failed":
            skipped_gold.append({"chunk_id": request.chunk_id, "reason": "gold_failed",
                                 "error_type": str(row.error.get("type") or "UnknownError")})
    fingerprint = digest({"manifest": manifest, "rows": [row.model_dump(mode="json") for row in rows.values()]})
    return requests, reference, fingerprint, skipped_gold
