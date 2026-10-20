"""Select paragraphs, generate reasoning-model gold, compare models, or report offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from graphrag_pipeline.extraction_benchmark.artifacts import load_gold
from graphrag_pipeline.extraction_benchmark.providers import EFFORTS, ModelSettings, parse_model
from graphrag_pipeline.extraction_benchmark.records import DEFAULT_DATASET, load_paragraphs
from graphrag_pipeline.extraction_benchmark.runner import report, run_extraction


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select", help="Preview the paragraph cohort without services or output files.")
    gold = sub.add_parser("gold", help="Generate/resume gold labels using an explicit OpenRouter reasoning model.")
    run = sub.add_parser("run", help="Compare OpenRouter/Ollama candidates with complete saved gold.")
    offline = sub.add_parser("report", help="Validate saved artifacts and regenerate summary.json offline.")
    offline.add_argument("--input", type=Path, required=True, help="Gold or candidate run directory.")
    for command in (select, gold):
        command.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
        command.add_argument("--paragraphs", type=int, default=150, help="Exact paragraph count, default 150.")
    gold.add_argument("--model", required=True, help="OpenRouter model ID, without the openrouter: prefix.")
    gold.add_argument("--reasoning-effort", choices=EFFORTS, required=True, help="Enabled gold reasoning effort; checked against advertised model support.")
    run.add_argument("--gold", type=Path, required=True, help="Complete gold-generation directory; original dataset is unnecessary.")
    run.add_argument("--model", action="append", required=True, help="Repeat provider:model, e.g. ollama:qwen3:4b or openrouter:vendor/model.")
    run.add_argument("--reasoning-effort", choices=("none", *EFFORTS), help="Candidate OpenRouter effort; omit to use model defaults.")
    run.add_argument("--ollama-think", default="default", help="default (omit), true, false, or a model-supported string such as high.")
    for command in (gold, run):
        command.add_argument("--output", type=Path, required=True, help="New/empty run directory, or existing compatible directory with --resume.")
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


def execute(args):
    if args.command == "report":
        summary = report(args.input)
        return (0 if summary["successful"] else 1), {"output": str(args.input), **summary}
    if args.command in {"select", "gold"}:
        requests = load_paragraphs(args.dataset_path, count=args.paragraphs)
    else:
        requests, reference, fingerprint = load_gold(args.gold)
        if args.output.resolve() == args.gold.resolve():
            raise ValueError("candidate output must differ from the gold directory")
    preview = {"paragraphs": len(requests), "dataset_sha256": requests[0].provenance.source_sha256,
               "records": len({r.record_id for r in requests}),
               "selection": [{"chunk_id": r.chunk_id, "record_id": r.record_id,
                              "paragraph_idx": r.paragraph_idx, "source_line": r.provenance.source_line,
                              "title": r.title} for r in requests]}
    if args.command == "select":
        return 0, preview
    if args.retry_failed and not args.resume:
        raise ValueError("--retry-failed requires --resume")
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
    if args.dry_run:
        return 0, {**preview, "dry_run": True, "models": [s.public() for s in settings]}
    summary = run_extraction(requests, settings, args.output,
        kind="gold" if args.command == "gold" else "candidate",
        reference=reference if args.command == "run" else None,
        gold_fingerprint=fingerprint if args.command == "run" else None,
        resume=args.resume, retry_failed=args.retry_failed, show_progress=not args.no_progress)
    return (0 if summary["successful"] else 1), {"output": str(args.output), **summary}


def main():
    args = build_parser().parse_args()
    load_dotenv()
    try:
        code, result = execute(args)
    except KeyboardInterrupt:
        code, result = 130, {"status": "interrupted", "message": "Completed rows remain saved; resume with the same configuration."}
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
