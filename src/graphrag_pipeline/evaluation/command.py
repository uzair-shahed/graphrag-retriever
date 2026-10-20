"""Prepare/check/run paired MuSiQue ans-dev evaluation, or report saved results offline."""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from graphrag_pipeline.answer import build_answer_llm
from graphrag_pipeline.config import Neo4jSettings, OllamaSettings, OpenRouterSettings, model_settings_from_env
from graphrag_pipeline.diagnostics import exception_diagnostics
from graphrag_pipeline.evaluation.providers import EvaluatorSettings, build_metrics
from graphrag_pipeline.evaluation.records import DEFAULT_DATASET, load_cases
from graphrag_pipeline.evaluation.report import generate_report
from graphrag_pipeline.evaluation.runner import run_evaluation, write_json
from graphrag_pipeline.evaluation.runtime import EvaluationSettings, PairedRetriever, graph_readiness, graph_snapshot
from graphrag_pipeline.ingestion.load_musique_dataset import load_selected_records
from graphrag_pipeline.retrieval_prerequisites import VectorIndexSettings, verify_retrieval_prerequisites
from graphrag_pipeline.storage.neo4j_connection import Neo4jConnection


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    report = subcommands.add_parser("report", help="Generate offline report.html from saved evaluation artifacts.")
    report.add_argument("--input", type=Path, required=True, help="Existing evaluation directory; replaces only its report.html.")
    for name in ("prepare", "check", "run"):
        command = subcommands.add_parser(name)
        command.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
        selectors = command.add_mutually_exclusive_group()
        selectors.add_argument("--record-id", action="append", help="Exact ID; repeat for arbitrary records.")
        selectors.add_argument("--records", type=int, dest="records", help="Number of records, including --start-record.")
        selectors.add_argument("--max-records", type=int, dest="records", help="Alias for --records.")
        command.add_argument("--start-record", type=int, help="1-indexed nonblank record position; default 1.")
        command.add_argument("--dry-run", action="store_true", help="Validate and print selection without Neo4j or model calls.")
        if name == "prepare":
            command.add_argument("--manifest-path", type=Path)
            command.add_argument("--no-progress", action="store_true")
        else:
            command.add_argument("--condition", action="append", choices=("record", "corpus"), help="Repeat; defaults to both.")
        if name == "run":
            command.add_argument("--no-progress", action="store_true", help="Disable the tqdm evaluation progress bar; final JSON is still printed.")
            command.add_argument("--output", type=Path, help="New artifact directory under runs/ by default.")
            command.add_argument("--top-k", type=int, default=5)
            command.add_argument("--max-context-chunks", type=int, default=5)
            command.add_argument("--max-context-tokens", type=int, default=4096)
            command.add_argument("--max-graph-nodes", type=int, default=200)
            command.add_argument("--answer-provider", choices=("ollama", "openrouter"), help="Answer provider only; retrieval embeddings retain GRAPH_RAG_PROVIDER.")
            command.add_argument("--answer-model")
            command.add_argument("--eval-provider", choices=("ollama", "openrouter"))
            command.add_argument("--eval-model")
            command.add_argument("--eval-embedding-model")
    return parser


def selection_from_args(args):
    if args.record_id and args.start_record is not None:
        raise ValueError("--record-id cannot be combined with --start-record")
    return load_cases(args.dataset_path, record_ids=args.record_id,
                      start_record=args.start_record if args.start_record is not None else 1, records=args.records)


def public_model_settings(settings):
    return {k: v for k, v in asdict(settings).items() if k != "api_key"}


async def execute(args):
    if args.command == "report":
        return 0, generate_report(args.input)
    cases = selection_from_args(args)
    selection = {"dataset": str(args.dataset_path), "dataset_sha256": cases[0].source_sha256,
                 "records_selected": len(cases), "records": [{"id": c.gold.id, "source_line": c.source_line} for c in cases]}
    if args.dry_run:
        return 0, {**selection, "dry_run": True, "chunks_planned": sum(len(c.gold.paragraphs) for c in cases)}
    if args.command == "prepare":
        summary = await load_selected_records([c.query(str(uuid.uuid4())) for c in cases],
            show_progress=not args.no_progress, manifest_path=args.manifest_path)
        payload = summary.to_dict()
        failed = summary.chunks_failed or summary.dead_lettered_records or summary.chunks_needs_investigation
        return (1 if failed or summary.records_succeeded != len(cases) else 0), payload
    conditions = tuple(args.condition or ("record", "corpus"))
    # Check validity before any network calls.
    settings = EvaluationSettings(conditions=conditions) if args.command == "check" else EvaluationSettings(
        args.top_k, args.max_context_chunks, args.max_context_tokens, args.max_graph_nodes, conditions)
    neo4j = Neo4jSettings.from_env()
    retrieval_models = model_settings_from_env()
    with Neo4jConnection(neo4j) as connection:
        ready = graph_readiness(connection, cases)
        if not ready["ready"]:
            return 2, {**selection, "preflight": ready, "action": "Run the explicit prepare command for missing records; inspect conflicts before ingestion."}
        index = VectorIndexSettings.from_env()
        prerequisites = verify_retrieval_prerequisites(
            connection, neo4j, model_settings=retrieval_models, vector_index=index,
            require_record_filter="record" in conditions, record_ids=[c.gold.id for c in cases] if "record" in conditions else (),
        ).to_dict()
        if args.command == "check":
            return 0, {**selection, "preflight": ready, "prerequisites": prerequisites}
        provider_factories = {"ollama": OllamaSettings.from_env, "openrouter": OpenRouterSettings.from_env}
        answer_models = provider_factories[args.answer_provider]() if args.answer_provider else retrieval_models
        if args.answer_model:
            answer_models = replace(answer_models, llm_model=args.answer_model)
        evaluator = EvaluatorSettings.from_env(args.eval_provider)
        if args.eval_model:
            evaluator = replace(evaluator, llm_model=args.eval_model)
        if args.eval_embedding_model:
            evaluator = replace(evaluator, embedding_model=args.eval_embedding_model)
        output = args.output or Path("runs/evaluation") / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8])
        if output.exists() and any(output.iterdir()):
            raise ValueError("output directory must be empty; choose a new directory")
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "preflight.json", {"selection": selection, "graph": ready, "prerequisites": prerequisites})
        retriever = PairedRetriever(connection.connect(), neo4j.database, retrieval_models, settings, index)
        metrics, client = build_metrics(evaluator)
        try:
            rows, summary = await run_evaluation(cases, retriever, build_answer_llm(answer_models), metrics, settings, output,
                configuration={"retrieval_models": public_model_settings(retrieval_models),
                    "answer_models": public_model_settings(answer_models), "evaluator": evaluator.public(),
                    "index": asdict(index), "neo4j_database": neo4j.database,
                    "query_policy": "original question only; no title hints or gold decomposition",
                    "answer_policy": "shared generator and validator for vector and graph-expanded evidence"},
                snapshot=lambda: graph_snapshot(connection, neo4j.database), metric_timeout=evaluator.timeout,
                show_progress=not args.no_progress)
        finally:
            await client.close()
        failed = (summary["snapshot_consistent"] is not True or
                  any(r["response"]["status"] == "failed" or not r["paired_seeds_consistent"] or not r["graph_component_complete"] or
                      any(s["status"] == "failed" for s in r["ragas"].values()) or
                      r["retrieval_context_relevance"]["status"] == "failed" for r in rows))
        return (1 if failed else 0), {"output": str(output), **summary}


def main():
    args = build_parser().parse_args()
    try:
        code, result = asyncio.run(execute(args))
    except KeyboardInterrupt:
        code, result = 130, {"status": "interrupted", "message": "Completed preparation checkpoints or evaluation rows remain available."}
    except Exception as error:
        code, result = 2, exception_diagnostics(error)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
