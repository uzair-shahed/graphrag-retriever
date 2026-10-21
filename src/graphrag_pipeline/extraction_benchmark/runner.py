"""Resumable extraction runs and offline summaries, separate from RAGAS."""

from __future__ import annotations

import time
import uuid
from contextlib import ExitStack, nullcontext
from pathlib import Path

from tqdm import tqdm

from graphrag_pipeline.extraction_benchmark.artifacts import (
    FORMAT_VERSION, SavedExtraction, commit_append, digest, load_run, now, recover_append, run_lock,
    validate_gold_coverage, write_json, write_jsonl,
)
from graphrag_pipeline.extraction_benchmark.contracts import ParagraphExtraction, validate_extraction_response
from graphrag_pipeline.extraction_benchmark.providers import IncompleteGeneration, JSONProvider, SYSTEM_PROMPT
from graphrag_pipeline.extraction_benchmark.scoring import SCORING_VERSION, aggregate, score


def summarize(manifest, requests, rows) -> dict:
    models = {}
    candidate = manifest["kind"] == "candidate"
    skipped_gold = manifest["configuration"].get("skipped_gold", []) if candidate else []
    expected = len(requests) - len(skipped_gold)
    for settings in manifest["configuration"]["models"]:
        spec = f"{settings['provider']}:{settings['model']}"
        selected = [row for row in rows.values() if row.model == spec]
        models[spec] = {
            "expected": expected, "attempted": len(selected),
            "succeeded": sum(row.status == "succeeded" for row in selected),
            "failed": sum(row.status == "failed" for row in selected),
            "pending": expected - len(selected),
            "elapsed_seconds": sum(row.elapsed_seconds for row in selected),
            "generation_formats": {
                "legacy_v1": sum(row.response_format_version == 1 for row in selected),
                "simplified_v2": sum(row.response_format_version == 2 for row in selected),
            },
        }
        if candidate:
            models[spec]["compared"] = len(selected)
            models[spec]["micro"] = aggregate([row.scores for row in selected])
    summary = {"format_version": FORMAT_VERSION, "kind": manifest["kind"], "run_id": manifest["run_id"],
            "complete": all(model["pending"] == 0 for model in models.values()),
            "successful": all(model["pending"] == 0 and model["failed"] == 0 for model in models.values()),
            "score_population": "completed rows, including failed extractions as empty predictions; pending rows not scored",
            "models": models}
    if candidate:
        summary.update(requested=len(requests), comparable=expected, skipped=len(skipped_gold),
                       skipped_gold=skipped_gold,
                       score_population="completed candidate rows with usable gold, including failed extractions as empty predictions; pending rows and skipped gold not scored")
    return summary


def report(directory: Path):
    if not directory.is_dir():
        raise ValueError("report input must be an existing benchmark directory")
    with run_lock(directory):
        recover_append(directory)
        manifest, requests, rows, _ = load_run(directory)
        summary = summarize(manifest, requests, rows)
        write_json(directory / "summary.json", summary)
    return summary


def run_extraction(requests, settings, output: Path, *, kind: str, reference=None,
                   gold_fingerprint=None, skipped_gold=None, resume=False, retry_failed=False,
                   append_start=None, dry_run=False, show_progress=True, provider_factory=JSONProvider):
    """Persist each attempt atomically; retries of saved failures are explicit."""
    if retry_failed and not resume:
        raise ValueError("--retry-failed requires --resume")
    if kind == "gold" and (len(settings) != 1 or settings[0].provider != "openrouter" or settings[0].reasoning_effort in {None, "none"}):
        raise ValueError("gold needs one OpenRouter model and an explicit enabled reasoning effort")
    if kind not in {"gold", "candidate"} or not requests or not settings:
        raise ValueError("run needs a valid kind, nonempty cohort and model list")
    if kind == "candidate" and reference is None:
        raise ValueError("candidate run needs a gold reference mapping")
    if len({request.chunk_id for request in requests}) != len(requests):
        raise ValueError("duplicate paragraph IDs in the requested cohort")
    if reference is not None:
        # Callers may supply references for a larger gold cohort.
        reference = {r.chunk_id: reference[r.chunk_id] for r in requests if r.chunk_id in reference}
        skipped_gold = validate_gold_coverage(requests, reference, skipped_gold or [])
    elif skipped_gold:
        raise ValueError("skipped gold is only valid for candidate runs")
    if append_start is not None and not (output / "manifest.json").is_file():
        raise ValueError("append requires an existing benchmark run")
    specs = [setting.spec for setting in settings]
    if len(set(specs)) != len(specs):
        raise ValueError("duplicate provider/model entries; use separate directories for different settings")
    cohort = [request.model_dump(mode="json") for request in requests]
    reference_rows = [reference[request.chunk_id].model_dump(mode="json") for request in requests
                      if request.chunk_id in reference] if reference is not None else None
    config = {"paragraphs": len(requests), "cohort_sha256": digest(cohort),
              "models": [setting.public() for setting in settings], "prompt": SYSTEM_PROMPT,
              "response_schema": ParagraphExtraction.model_json_schema(), "scoring_version": SCORING_VERSION}
    if reference_rows is not None:
        config.update(gold_fingerprint=gold_fingerprint, reference_sha256=digest(reference_rows))
        if skipped_gold:
            config["skipped_gold"] = skipped_gold
    with (nullcontext() if dry_run else run_lock(output)):
        occupied = [path for path in output.iterdir() if path.name != ".benchmark.lock"] if output.exists() else []
        extension = None
        if occupied:
            if not resume and append_start is None:
                raise ValueError("output directory is not empty; use --resume with identical settings or a new directory")
            manifest, saved_requests, rows, saved_reference = load_run(output)
            saved_count = manifest["configuration"]["paragraphs"]
            if append_start is not None:
                if append_start != saved_count + 1 or len(requests) <= saved_count:
                    raise ValueError(f"append range must start at paragraph {saved_count + 1} and extend the saved cohort")
                mutable = {"paragraphs", "cohort_sha256", "reference_sha256", "gold_fingerprint", "skipped_gold"}
                saved_fixed = {key: value for key, value in manifest["configuration"].items() if key not in mutable}
                new_fixed = {key: value for key, value in config.items() if key not in mutable}
                if manifest["kind"] != kind or saved_fixed != new_fixed or saved_requests != requests[:saved_count]:
                    raise ValueError("append configuration/source mismatch; saved cohort must match the unchanged source prefix")
                if saved_reference is not None:
                    saved_ids = {r.chunk_id for r in saved_requests}
                    prefix_skips = [skip for skip in skipped_gold if skip["chunk_id"] in saved_ids]
                    if (any(saved_reference.get(r.chunk_id) != reference.get(r.chunk_id) for r in saved_requests)
                            or manifest["configuration"].get("skipped_gold", []) != prefix_skips):
                        raise ValueError("append gold reference mismatch; existing gold labels must be unchanged, including skipped gold")
                saved_expected = saved_count - len(manifest["configuration"].get("skipped_gold", []))
                if len(rows) != saved_expected * len(settings):
                    raise ValueError("saved run has pending paragraphs; use --resume to finish its cohort before appending")
                extension = manifest
                manifest = {**manifest, "configuration": config}
            elif saved_count != len(requests):
                raise ValueError(
                    f"resume paragraph count mismatch: saved run has {saved_count} paragraphs, "
                    f"but requested {len(requests)}. Resume with --paragraphs {saved_count} "
                    "to continue the saved cohort (including --retry-failed to retry failures); "
                    "use --additional-paragraphs or --start-paragraph/--end-paragraph to extend this run."
                )
            elif manifest["kind"] != kind or manifest["configuration"] != config or saved_requests != requests:
                raise ValueError("resume configuration/source/gold mismatch; choose a new output directory")
        else:
            if append_start is not None:
                raise ValueError("append requires an existing benchmark run")
            if resume:
                raise ValueError("cannot resume an empty directory")
            manifest = {"format_version": FORMAT_VERSION, "kind": kind, "run_id": uuid.uuid4().hex,
                        "created_at": now(), "configuration": config}
            rows = {}
        if dry_run:
            return summarize(manifest, requests, rows)
        recover_append(output)
        if extension is not None:
            commit_append(output, extension, manifest, cohort, reference_rows)
        elif not occupied:
            write_jsonl(output / "cohort.jsonl", cohort)
            if reference_rows is not None:
                write_jsonl(output / "reference.jsonl", reference_rows)
            write_json(output / "manifest.json", manifest)
        result_path = output / ("gold.jsonl" if kind == "gold" else "predictions.jsonl")
        def persist():
            write_jsonl(result_path, [row.model_dump(mode="json") for row in rows.values()])
            write_json(output / "summary.json", summarize(manifest, requests, rows))
        persist()
        eligible = [request for request in requests if reference is None or request.chunk_id in reference]
        pending = [(setting, request) for setting in settings for request in eligible
                   if (setting.spec, request.chunk_id) not in rows or
                   (retry_failed and rows[(setting.spec, request.chunk_id)].status == "failed")]
        # No service calls or credentials are needed when a resumed run has no work.
        with ExitStack() as stack:
            clients = {}
            preflight = {}
            for setting in settings:
                if any(item.spec == setting.spec for item, _ in pending):
                    client = stack.enter_context(provider_factory(setting))
                    preflight[setting.spec] = client.check()
                    clients[setting.spec] = client
            if preflight:
                write_json(output / "preflight.json", {"checked_at": now(), "models": preflight})
            progress = stack.enter_context(tqdm(total=len(pending), disable=not show_progress, desc=f"Extraction {kind}", unit="paragraph"))
            try:
                for setting, request in pending:
                    progress.set_postfix_str(f"{setting.spec} {request.chunk_id}")
                    key = (setting.spec, request.chunk_id)
                    previous = rows.get(key)
                    generation = None
                    error = None
                    result = None
                    started = time.monotonic()
                    try:
                        generation = clients[setting.spec].generate(request)
                        result = validate_extraction_response(request, generation.content)
                    except Exception as failure:
                        if isinstance(failure, IncompleteGeneration):
                            generation = failure.generation
                        message = str(failure)
                        if setting.api_key:
                            message = message.replace(setting.api_key, "[redacted]")
                        error = {"type": type(failure).__name__, "message": message}
                    rows[key] = SavedExtraction(
                        model=setting.spec, chunk_id=request.chunk_id,
                        status="failed" if error else "succeeded", result=result,
                        raw_content=generation.content if generation else None, error=error,
                        attempts=(previous.attempts if previous else 0) + 1,
                        timestamp=now(), elapsed_seconds=time.monotonic() - started,
                        metadata=generation.metadata if generation else {},
                        scores=score(reference[request.chunk_id], result) if reference is not None else None,
                    )
                    persist()
                    progress.update()
            finally:
                # KeyboardInterrupt leaves previous completed rows intact.
                persist()
        return summarize(manifest, requests, rows)
