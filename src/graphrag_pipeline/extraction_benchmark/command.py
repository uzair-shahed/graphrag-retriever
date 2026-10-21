"""Select paragraphs, generate reasoning-model gold, compare models, or report offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from graphrag_pipeline.extraction_benchmark.artifacts import load_gold, load_run
from graphrag_pipeline.extraction_benchmark.providers import EFFORTS, ModelSettings, parse_model
from graphrag_pipeline.extraction_benchmark.records import DEFAULT_DATASET, load_paragraphs
from graphrag_pipeline.extraction_benchmark.runner import report, run_extraction


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select", help="Preview the paragraph cohort without services or output files.")
    gold = sub.add_parser("gold", help="Generate/resume gold labels using an explicit OpenRouter reasoning model.")
    run = sub.add_parser("run", help="Compare OpenRouter/Ollama candidates with usable saved gold; report missing labels.")
    offline = sub.add_parser("report", help="Validate saved artifacts and regenerate summary.json offline.")
    offline.add_argument("--input", type=Path, required=True, help="Gold or candidate run directory.")
    for command in (select, gold):
        command.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
        command.add_argument("--paragraphs", type=int, default=150 if command is select else None,
                             help="Exact paragraph prefix count, default 150; cannot be combined with append selectors.")
    gold.add_argument("--model", required=True, help="OpenRouter model ID, without the openrouter: prefix.")
    gold.add_argument("--reasoning-effort", choices=EFFORTS, required=True, help="Enabled gold reasoning effort; checked against advertised model support.")
    run.add_argument("--gold", type=Path, required=True, help="Gold-generation directory; failed/missing labels are skipped and reported.")
    run.add_argument("--model", action="append", required=True, help="Repeat provider:model, e.g. ollama:qwen3:4b or openrouter:vendor/model.")
    run.add_argument("--reasoning-effort", choices=("none", *EFFORTS), help="Candidate OpenRouter effort; omit to use model defaults.")
    run.add_argument("--ollama-think", default="default", help="default (omit), true, false, or a model-supported string such as high.")
    for command in (gold, run):
        command.add_argument("--output", type=Path, required=True, help="New/empty run directory, or existing compatible directory with --resume or append selectors.")
        append = command.add_mutually_exclusive_group()
        append.add_argument("--additional-paragraphs", type=int, help="Append the next N paragraphs to an existing run.")
        append.add_argument("--start-paragraph", type=int, help="First new paragraph's 1-based source-order position; requires --end-paragraph.")
        command.add_argument("--end-paragraph", type=int, help="Last new paragraph's inclusive source-order position; requires --start-paragraph.")
        command.add_argument("--max-tokens", type=int, default=16384, help="Output-token budget (including reasoning where applicable), default 16384.")
        command.add_argument("--timeout", type=float, default=180, help="HTTP request timeout in seconds, default 180.")
        command.add_argument("--retries", type=int, default=2, help="Additional transient HTTP/transport retries, default 2.")
        command.add_argument("--openrouter-base-url", help="Override OPENROUTER_BASE_URL; default https://openrouter.ai/api/v1.")
        command.add_argument("--resume", action="store_true", help="Validate the existing run and skip saved attempts.")
        command.add_argument("--retry-failed", action="store_true", help="With --resume, repeat failed saved extractions; successful ones stay skipped.")
        command.add_argument("--no-progress", action="store_true", help="Hide stderr progress; JSON summary still prints to stdout.")
        command.add_argument("--dry-run", action="store_true", help="Validate local selection/configuration without services or file writes.")
    run.add_argument("--ollama-host", help="Override OLLAMA_HOST; default http://localhost:11434 (native API, no /v1).")
    return parser


def append_range(args):
    """Resolve an explicit append selector against the saved cohort, read-only."""
    additional = args.additional_paragraphs
    start, end = args.start_paragraph, args.end_paragraph
    if additional is None and start is None and end is None:
        return None
    if getattr(args, "paragraphs", None) is not None:
        raise ValueError("--paragraphs cannot be combined with append selectors")
    if additional is not None:
        if start is not None or end is not None or additional < 1:
            raise ValueError("--additional-paragraphs must be positive and cannot be combined with a range")
    elif start is None or end is None or start < 1 or end < start:
        raise ValueError("append range requires positive --start-paragraph and --end-paragraph with end >= start")
    if not (args.output / "manifest.json").is_file():
        raise ValueError("append requires an existing benchmark run")
    manifest, requests, rows, _ = load_run(args.output)
    count = len(requests)
    expected = count - len(manifest["configuration"].get("skipped_gold", []))
    if len(rows) != expected * len(manifest["configuration"]["models"]):
        raise ValueError("saved run has pending paragraphs; use --resume to finish its cohort before appending")
    if additional is not None:
        start, end = count + 1, count + additional
    if start != count + 1:
        raise ValueError(f"append range must start at paragraph {count + 1}; overlaps and gaps are not allowed")
    return start, end


def execute(args):
    if args.command == "report":
        summary = report(args.input)
        return (0 if summary["successful"] else 1), {"output": str(args.input), **summary}
    if args.command != "select" and args.retry_failed and not args.resume:
        raise ValueError("--retry-failed requires --resume")
    extension = append_range(args) if args.command != "select" else None
    if args.command in {"select", "gold"}:
        count = extension[1] if extension else (150 if args.paragraphs is None else args.paragraphs)
        requests = load_paragraphs(args.dataset_path, count=count)
    else:
        if args.output.resolve() == args.gold.resolve():
            raise ValueError("candidate output must differ from the gold directory")
        requests, reference, fingerprint, skipped_gold = load_gold(args.gold)
        count = extension[1] if extension else len(requests)
        if args.resume and extension is None:
            _, saved_requests, _, _ = load_run(args.output)
            count = len(saved_requests)
        if count > len(requests):
            raise ValueError(f"gold cohort contains only {len(requests)} paragraphs; requested {count}")
        requests = requests[:count]
        reference = {request.chunk_id: reference[request.chunk_id] for request in requests if request.chunk_id in reference}
        selected_ids = {request.chunk_id for request in requests}
        skipped_gold = [skip for skip in skipped_gold if skip["chunk_id"] in selected_ids]
    preview = {"paragraphs": len(requests), "dataset_sha256": requests[0].provenance.source_sha256,
               "records": len({r.record_id for r in requests}),
               "selection": [{"chunk_id": r.chunk_id, "record_id": r.record_id,
                              "paragraph_idx": r.paragraph_idx, "source_line": r.provenance.source_line,
                              "title": r.title} for r in requests]}
    if args.command == "select":
        return 0, preview
    if extension:
        preview["append"] = {"previous_paragraphs": extension[0] - 1,
                             "additional_paragraphs": extension[1] - extension[0] + 1,
                             "start_paragraph": extension[0], "end_paragraph": extension[1]}
    specs = ["openrouter:" + args.model] if args.command == "gold" else args.model
    if len(set(specs)) != len(specs):
        raise ValueError("duplicate provider/model entries")
    for spec in specs:
        parse_model(spec)
    settings = [ModelSettings.from_env(spec, max_tokens=args.max_tokens, timeout=args.timeout,
                retries=args.retries, reasoning_effort=args.reasoning_effort,
                ollama_think=getattr(args, "ollama_think", "default"),
                openrouter_base_url=args.openrouter_base_url, ollama_host=getattr(args, "ollama_host", None))
                for spec in specs]
    summary = run_extraction(requests, settings, args.output,
        kind="gold" if args.command == "gold" else "candidate",
        reference=reference if args.command == "run" else None,
        gold_fingerprint=fingerprint if args.command == "run" else None,
        skipped_gold=skipped_gold if args.command == "run" else None,
        resume=args.resume, retry_failed=args.retry_failed, show_progress=not args.no_progress,
        append_start=extension[0] if extension else None, dry_run=args.dry_run)
    if args.dry_run:
        if args.command == "run":
            preview.update({key: summary[key] for key in ("requested", "comparable", "skipped", "skipped_gold")})
        return 0, {**preview, "dry_run": True, "models": [s.public() for s in settings]}
    return (0 if summary["successful"] else 1), {"output": str(args.output), **summary}


def main():
    args = build_parser().parse_args()
    load_dotenv()
    try:
        code, result = execute(args)
    except KeyboardInterrupt:
        code, result = 130, {"status": "interrupted", "message": (
            "Completed rows remain saved; use --resume without append selectors. "
            "For gold, set --paragraphs to the saved manifest's paragraph count."
        )}
    except Exception as error:
        # Provider HTTP errors omit credentials; do not dump configuration/headers.
        import os
        message = str(error)
        secret = os.getenv("OPENROUTER_API_KEY")
        if secret:
            message = message.replace(secret, "[redacted]")
        code, result = 2, {"status": "failed", "type": type(error).__name__, "error": message}
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
