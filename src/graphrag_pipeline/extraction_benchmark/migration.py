"""Offline v1-to-v2 projection of saved generations; never contacts providers."""

from __future__ import annotations

import hashlib
import tempfile
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path

from graphrag_pipeline.extraction.schemas import ExtractionRequest, ExtractionResult, validate_result_for_request
from graphrag_pipeline.extraction_benchmark.artifacts import (
    APPEND_TRANSACTION, FORMAT_VERSION, SavedExtraction, digest, load_gold, load_run, now,
    read_json, read_jsonl, run_lock, validate_gold_coverage, validate_rows, write_json, write_jsonl,
)
from graphrag_pipeline.extraction_benchmark.contracts import (
    ParagraphExtraction, project_legacy_payload, validate_extraction_response,
)
from graphrag_pipeline.extraction_benchmark.providers import ModelSettings, SYSTEM_PROMPT
from graphrag_pipeline.extraction_benchmark.runner import summarize
from graphrag_pipeline.extraction_benchmark.scoring import SCORING_VERSION, score


def _file_hashes(directory):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir())
            if p.is_file() and p.name != ".benchmark.lock"}


def _legacy_result(raw, request):
    result = ExtractionResult.model_validate(project_legacy_payload(raw))
    return validate_result_for_request(request, result)


def _read_legacy(directory):
    if (directory / APPEND_TRANSACTION).exists():
        raise ValueError("finish the original append transaction with the v1 implementation before migrating")
    manifest = read_json(directory / "manifest.json")
    if manifest.get("format_version") != 1 or manifest.get("kind") not in {"gold", "candidate"}:
        raise ValueError("migrate input must be a format-v1 gold or candidate run")
    config = manifest["configuration"]
    if config.get("scoring_version") != SCORING_VERSION:
        raise ValueError("unsupported source scoring version")
    requests = [ExtractionRequest.model_validate(r) for r in read_jsonl(directory / "cohort.jsonl")]
    cohort = [r.model_dump(mode="json") for r in requests]
    if (not requests or len(requests) != config["paragraphs"]
            or len({r.chunk_id for r in requests}) != len(requests)
            or digest(cohort) != config["cohort_sha256"]):
        raise ValueError("source cohort count/identity/checksum mismatch")
    settings = [ModelSettings(**s) for s in config["models"]]
    specs = [s.spec for s in settings]
    if not specs or len(specs) != len(set(specs)):
        raise ValueError("missing or duplicate source model configurations")
    reference = None
    if manifest["kind"] == "gold":
        if len(settings) != 1 or settings[0].provider != "openrouter" or settings[0].reasoning_effort in {None, "none"}:
            raise ValueError("source gold must use one OpenRouter model with reasoning enabled")
    else:
        raw_reference = read_jsonl(directory / "reference.jsonl")
        if digest(raw_reference) != config["reference_sha256"]:
            raise ValueError("source frozen reference checksum mismatch")
        skips = config.get("skipped_gold", [])
        skipped_ids = {s["chunk_id"] for s in skips}
        eligible = [r for r in requests if r.chunk_id not in skipped_ids]
        reference = {r.chunk_id: _legacy_result(raw, r)
                     for raw, r in zip(raw_reference, eligible, strict=True)}
        if validate_gold_coverage(requests, reference, skips) != skips:
            raise ValueError("source skipped gold must follow cohort order")
    path = directory / ("gold.jsonl" if manifest["kind"] == "gold" else "predictions.jsonl")
    raw_rows = read_jsonl(path) if path.exists() else []
    by_id = {r.chunk_id: r for r in requests}
    projected, audit_rows = [], {}
    for raw in raw_rows:
        key = (raw["model"], raw["chunk_id"])
        if key in audit_rows or raw["chunk_id"] not in by_id:
            raise ValueError("unknown paragraph or duplicate source row")
        request = by_id[raw["chunk_id"]]
        row = {**raw, "response_format_version": 1}
        if raw.get("result") is not None:
            row["result"] = _legacy_result(raw["result"], request).model_dump(mode="json")
        projected.append(row)
        audit_rows[key] = {"model": raw["model"], "chunk_id": raw["chunk_id"],
                           "status": raw["status"], "error": raw["error"],
                           "raw_content_sha256": digest(raw.get("raw_content"))}
    # Verify successful raw/result agreement and original scores before recovering failures.
    rows = validate_rows(projected, requests, specs, reference=reference, legacy_rows=audit_rows)
    return manifest, requests, rows, reference, list(audit_rows.values())


@contextmanager
def _destination_lock(output):
    lock = output.parent / f".{output.name}.migration.lock"
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError:
        raise ValueError(f"migration destination is locked: {lock}") from None
    try:
        with handle:
            handle.write("offline migration\n")
        yield
    finally:
        lock.unlink(missing_ok=True)


def migrate_run(source: Path, output: Path, *, gold: Path | None = None, dry_run=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    gold = Path(gold).resolve() if gold is not None else None
    if not source.is_dir():
        raise ValueError("migration input must be an existing run directory")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("migration output must be separate from the source, without nesting")
    if gold is not None and (not gold.is_dir() or gold == source or gold == output
                             or gold in output.parents or output in gold.parents):
        raise ValueError("converted gold must be an existing separate directory")
    sources = sorted({source, *([gold] if gold is not None else [])})
    with ExitStack() as stack:
        for directory in sources:
            if (directory / ".benchmark.lock").exists():
                raise ValueError(f"source is locked; finish its writer before migrating: {directory}")
            if not dry_run:
                stack.enter_context(run_lock(directory))
        if not dry_run:
            output.parent.mkdir(parents=True, exist_ok=True)
            stack.enter_context(_destination_lock(output))
        elif (output.parent / f".{output.name}.migration.lock").exists():
            raise ValueError("migration destination is locked")
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError("migration output must be a new or empty directory")
        source_hashes = _file_hashes(source)
        old_manifest, requests, old_rows, old_reference, audit_rows = _read_legacy(source)
        reference = None
        fingerprint = None
        skipped = []
        if old_manifest["kind"] == "candidate":
            if gold is None:
                raise ValueError("candidate migration requires --gold pointing to converted v2 gold")
            gold_requests, gold_reference, fingerprint, gold_skips = load_gold(gold)
            if requests != gold_requests[:len(requests)]:
                raise ValueError("converted gold cohort differs from the candidate source")
            if any(gold_reference.get(key) != result for key, result in old_reference.items()):
                raise ValueError("converted gold changes a previously frozen label")
            ids = {r.chunk_id for r in requests}
            reference = {key: value for key, value in gold_reference.items() if key in ids}
            skipped = [s for s in gold_skips if s["chunk_id"] in ids]
            validate_gold_coverage(requests, reference, skipped)
        elif gold is not None:
            raise ValueError("--gold is only used when migrating a candidate run")
        rows = {}
        recovered = 0
        for key, old in old_rows.items():
            request = next(r for r in requests if r.chunk_id == old.chunk_id)
            result, error = None, old.error
            incomplete = ((old.error or {}).get("type") == "IncompleteGeneration"
                          or old.metadata.get("finish_reason") not in {None, "stop"})
            if old.raw_content is not None and not incomplete:
                try:
                    result = validate_extraction_response(request, old.raw_content, response_format_version=1)
                    error = None
                except ValueError as failure:
                    error = {"type": type(failure).__name__, "message": str(failure)}
            if result is None and not error:
                raise ValueError("source has an unavailable/incomplete response without a saved failure")
            row = SavedExtraction.model_validate({**old.model_dump(mode="json"),
                "status": "succeeded" if result is not None else "failed",
                "result": result, "error": error,
                "scores": score(reference[old.chunk_id], result) if reference is not None else None})
            recovered += old.status == "failed" and row.status == "succeeded"
            rows[key] = row
        cohort = [r.model_dump(mode="json") for r in requests]
        config = {"paragraphs": len(requests), "cohort_sha256": digest(cohort),
                  "models": old_manifest["configuration"]["models"], "prompt": SYSTEM_PROMPT,
                  "response_schema": ParagraphExtraction.model_json_schema(), "scoring_version": SCORING_VERSION}
        reference_rows = None
        if reference is not None:
            reference_rows = [reference[r.chunk_id].model_dump(mode="json") for r in requests if r.chunk_id in reference]
            config.update(gold_fingerprint=fingerprint, reference_sha256=digest(reference_rows))
            if skipped:
                config["skipped_gold"] = skipped
        audit = {"source_directory": str(source), "source_manifest": old_manifest,
                 "source_file_sha256": source_hashes, "converted_at": now(), "rows": audit_rows}
        manifest = {"format_version": FORMAT_VERSION, "kind": old_manifest["kind"],
                    "run_id": uuid.uuid4().hex, "created_at": now(), "configuration": config,
                    "migration": {"source_run_id": old_manifest["run_id"], "audit_sha256": digest(audit),
                                  "recovered_rows": recovered, "migrated_rows": len(rows)}}
        summary = summarize(manifest, requests, rows)
        if _file_hashes(source) != source_hashes or (dry_run and any((p / ".benchmark.lock").exists() for p in sources)):
            raise ValueError("source changed during migration; wait for its writer to finish")
        conversion = {"recovered_rows": recovered, "migrated_rows": len(rows), "provider_calls": 0}
        if dry_run:
            return {**summary, "conversion": conversion, "dry_run": True}
        with tempfile.TemporaryDirectory(prefix=f".{output.name}.migrate-", dir=output.parent) as temporary:
            stage = Path(temporary)
            write_jsonl(stage / "cohort.jsonl", cohort)
            if reference_rows is not None:
                write_jsonl(stage / "reference.jsonl", reference_rows)
            write_jsonl(stage / ("gold.jsonl" if manifest["kind"] == "gold" else "predictions.jsonl"),
                        [r.model_dump(mode="json") for r in rows.values()])
            write_json(stage / "migration.json", audit)
            write_json(stage / "manifest.json", manifest)
            if (source / "preflight.json").exists():
                write_json(stage / "preflight.json", read_json(source / "preflight.json"))
            loaded_manifest, loaded_requests, loaded_rows, _ = load_run(stage)
            if summarize(loaded_manifest, loaded_requests, loaded_rows) != summary:
                raise ValueError("staged migration does not reproduce its expected summary")
            write_json(stage / "summary.json", summary)
            if output.exists():
                output.rmdir()  # Only an empty destination, checked under its migration lock.
            stage.rename(output)
        return {**summary, "conversion": conversion}
