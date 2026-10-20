"""Incremental paired artifacts, common answering, and explicit aggregation."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections import Counter
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from tqdm import tqdm

from graphrag_pipeline.answer import AnswerSettings, generate_answer
from graphrag_pipeline.answer_context import ANSWER_PROMPT_TEMPLATE, render_evidence_context
from graphrag_pipeline.diagnostics import exception_diagnostics
from graphrag_pipeline.query import QueryResponse
from graphrag_pipeline.retrieval import RetrievalResult
from graphrag_pipeline.evaluation.runtime import select_context
from graphrag_pipeline.evaluation.scoring import evidence_metrics, gold_metrics, triad_scores


def write_json(path: Path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


class CapturingLLM:
    def __init__(self, llm):
        self.llm = llm
        self.input = None
        self.usage = None

    def invoke(self, **kwargs):
        self.input = kwargs["input"]
        result = self.llm.invoke(**kwargs)
        usage = getattr(result, "usage", None)
        self.usage = usage.model_dump(mode="json") if hasattr(usage, "model_dump") else None
        return result


def _mean(values):
    values = [v for v in values if v is not None]
    return {"mean": sum(values) / len(values) if values else None, "count": len(values)}


def summarize(rows):
    groups = {}
    pairs = {}
    for row in rows:
        groups.setdefault(f'{row["condition"]}/{row["backend"]}', []).append(row)
        pairs.setdefault(row["pair_id"], {})[row["backend"]] = row
    summary = {}
    for group, values in groups.items():
        statuses = Counter(r["response"]["status"] for r in values)
        summary[group] = {
            "queries": len(values), "statuses": dict(statuses),
            "abstention_rate": statuses["insufficient_evidence"] / len(values),
            "failure_rate": statuses["failed"] / len(values),
            "traversal_incomplete": sum(not r["graph_component_complete"] for r in values),
            "metrics": {m: _mean(r["ragas"][m]["value"] for r in values) for m in ("faithfulness", "answer_relevancy", "context_relevance")},
            "metric_failures": {m: sum(r["ragas"][m]["status"] == "failed" for r in values) for m in ("faithfulness", "answer_relevancy", "context_relevance")},
            "answer_accuracy": _mean(float(r["gold"]["answer_correct"]) for r in values),
            "answer_f1": _mean(r["gold"]["answer_f1"] for r in values),
            "supporting_recall": _mean(r["retrieval_gold"]["supporting_recall"] for r in values),
            "supporting_precision": _mean(r["retrieval_gold"]["supporting_precision"] for r in values),
            "hop_coverage": _mean(r["retrieval_gold"]["hop_coverage"] for r in values),
            "answer_context_supporting_recall": _mean(r["answer_context_gold"]["supporting_recall"] for r in values),
            "citation_gold_support_precision": _mean(r["answer_context_gold"]["citation_gold_support_precision"] for r in values),
            "retrieval_context_relevance": _mean(r["retrieval_context_relevance"]["value"] for r in values),
            "retrieval_context_metric_failures": sum(r["retrieval_context_relevance"]["status"] == "failed" for r in values),
            "answerability_accuracy": _mean(float(r["gold"]["answerability_correct"]) for r in values),
            "context_tokens": _mean(r["context_tokens"] for r in values),
            "latency_seconds": _mean(r["timings"]["total_seconds"] for r in values),
        }
    deltas = {}
    for condition in {r["condition"] for r in rows}:
        matched = [p for p in pairs.values() if len(p) == 2 and p["vectorrag"]["condition"] == condition and all(r["paired_seeds_consistent"] for r in p.values())]
        deltas[condition] = {
            m: _mean(p["graphrag"]["ragas"][m]["value"] - p["vectorrag"]["ragas"][m]["value"] for p in matched if all(p[b]["ragas"][m]["value"] is not None for b in ("vectorrag", "graphrag")))
            for m in ("faithfulness", "answer_relevancy", "context_relevance")
        }
        deltas[condition]["answer_accuracy"] = _mean(float(p["graphrag"]["gold"]["answer_correct"]) - float(p["vectorrag"]["gold"]["answer_correct"]) for p in matched)
    return {"groups": summary, "paired_deltas_graphrag_minus_vectorrag": deltas, "completed_rows": len(rows)}


async def run_evaluation(cases, retriever, llm, metrics, settings, output, *,
                         configuration=None, snapshot=None, count_tokens=None, metric_timeout=180,
                         show_progress=True):
    """No ingestion dependencies: evaluating an existing graph is read-only."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(p.name != "preflight.json" for p in output.iterdir()):
        raise ValueError("output directory already contains evaluation artifacts; choose a new directory")
    if count_tokens is None:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        count_tokens = lambda text: len(encoding.encode(text, disallowed_special=()))
    run_id = str(uuid.uuid4())
    before = snapshot() if snapshot else None
    manifest = {
        "schema_version": 1, "run_id": run_id, "status": "running",
        "dataset": cases[0].source_file, "dataset_sha256": cases[0].source_sha256,
        "selected_records": [{"id": c.gold.id, "source_line": c.source_line} for c in cases],
        "settings": asdict(settings), "configuration": configuration or {},
        "versions": {p: version(p) for p in ("ragas", "neo4j-graphrag", "neo4j", "tiktoken")},
        "context_tokenizer": "cl100k_base (comparison proxy; not the answer model's native tokenizer)",
        "snapshot_before": before, "snapshot_after": None,
    }
    write_json(output / "manifest.json", manifest)
    rows = []
    stopped = False
    try:
        with (output / "records.jsonl").open("x", encoding="utf-8") as stream, tqdm(
            total=len(cases) * len(settings.conditions) * 2,
            desc="Evaluating", unit="result", disable=not show_progress,
        ) as progress:
            for index, case in enumerate(cases):
                progress.set_postfix(record=f"{index + 1}/{len(cases)}", phase="retrieving")
                embedding_error = None
                embedding_start = time.perf_counter()
                try:
                    vector = retriever.embed(case.gold.question)
                except Exception as error:
                    embedding_error = exception_diagnostics(error)
                    vector = None
                embedding_seconds = time.perf_counter() - embedding_start
                for condition in settings.conditions:
                    pair_id = f"{run_id}:{index}:{condition}"
                    retrieved, retrieval_errors, retrieval_seconds = {}, {}, {}
                    for backend in ("vectorrag", "graphrag"):
                        started = time.perf_counter()
                        try:
                            if embedding_error:
                                retrieval_errors[backend] = embedding_error
                                retrieved[backend] = []
                            else:
                                retrieved[backend] = retriever.retrieve(backend, vector, case.gold.id if condition == "record" else None)
                        except Exception as error:
                            retrieved[backend] = []
                            retrieval_errors[backend] = exception_diagnostics(error)
                        retrieval_seconds[backend] = time.perf_counter() - started
                    seeds_consistent = not retrieval_errors and [e.chunk_id for e in retrieved["vectorrag"]] == [e.chunk_id for e in retrieved["graphrag"]]
                    for backend in ("vectorrag", "graphrag"):
                        started = time.perf_counter()
                        trace_id = f"{pair_id}:{backend}"
                        progress.set_postfix(record=f"{index + 1}/{len(cases)}", condition=condition,
                                             backend=backend, phase="answering")
                        request = case.query(trace_id, backend)
                        full = retrieved[backend]
                        evidence = select_context(full, settings, count_tokens)
                        mode = "vector" if backend == "vectorrag" else "vector_cypher"
                        capturing = CapturingLLM(llm)
                        error = retrieval_errors.get(backend)
                        # Do not answer mismatched seeds as if this were a controlled pair.
                        if error or (not retrieval_errors and not seeds_consistent):
                            response = QueryResponse(
                                **{k: v for k, v in request.model_dump().items() if k in QueryResponse.model_fields and k != "retrieval_backend"},
                                status="failed", retrieval_mode=mode, retrieval_backend=backend, evidence=evidence,
                                error=json.dumps(error) if error else "paired vector seeds differ; comparison is invalid",
                            )
                        else:
                            response = generate_answer(request, RetrievalResult(trace_id, request.question, mode, evidence, backend),
                                capturing, answer_settings=AnswerSettings(mode, settings.max_context_chunks))
                        answer_seconds = time.perf_counter() - started
                        scoring_started = time.perf_counter()
                        progress.set_postfix(record=f"{index + 1}/{len(cases)}", condition=condition,
                                             backend=backend, phase="scoring")
                        scores = await triad_scores(metrics, response, evidence, timeout=metric_timeout)
                        if full == evidence:
                            retrieval_score = scores["context_relevance"]
                        else:
                            retrieval_score = (await triad_scores({"context_relevance": metrics["context_relevance"]}, response, full, timeout=metric_timeout))["context_relevance"]
                        retrieval_gold = evidence_metrics(case, full)
                        added = set(evidence_metrics(case, retrieved["graphrag"])["supporting_covered"]) - set(evidence_metrics(case, retrieved["vectorrag"])["supporting_covered"])
                        row = {
                            "run_id": run_id, "pair_id": pair_id, "trace_id": trace_id,
                            "record_id": case.gold.id, "source_line": case.source_line,
                            "backend": backend, "condition": condition,
                            "request": request.model_dump(mode="json"),
                            "labels": case.gold.model_dump(mode="json"),
                            "retrieval_evidence": [e.model_dump(mode="json") for e in full],
                            "answer_evidence": [e.model_dump(mode="json") for e in evidence],
                            "answer_context": render_evidence_context(evidence),
                            "answer_prompt": capturing.input,
                            "response": response.model_dump(mode="json"),
                            "graph_component_complete": all(e.graph_component_complete for e in full),
                            "context_selected": full != evidence,
                            "context_tokens": count_tokens(render_evidence_context(evidence)),
                            "context_bytes": len(render_evidence_context(evidence).encode()),
                            "seed_chunk_ids": [e.chunk_id for e in full],
                            "paired_seeds_consistent": seeds_consistent,
                            "graph_added_supporting_chunk_ids": sorted(added) if seeds_consistent else None,
                            "gold": gold_metrics(case, response), "retrieval_gold": retrieval_gold,
                            "answer_context_gold": evidence_metrics(case, evidence, response.citations),
                            "ragas": scores, "retrieval_context_relevance": retrieval_score,
                            "usage": capturing.usage, "provider_cost": None,
                            "timings": {"embedding_seconds": embedding_seconds, "retrieval_seconds": retrieval_seconds[backend],
                                "answer_seconds": answer_seconds, "evaluation_seconds": time.perf_counter() - scoring_started,
                                "total_seconds": embedding_seconds + retrieval_seconds[backend] + answer_seconds},
                        }
                        # Reuse the existing diagnostics redaction for response error text.
                        if row["response"]["error"]:
                            row["response"]["error"] = exception_diagnostics(RuntimeError(row["response"]["error"]))["error"]
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                        # Retain only aggregate inputs in memory; full contexts/labels
                        # are durable in JSONL and can be large over the dev split.
                        rows.append({k: row[k] for k in (
                            "pair_id", "condition", "backend", "ragas", "gold", "retrieval_gold",
                            "graph_component_complete", "paired_seeds_consistent", "timings",
                            "answer_context_gold", "retrieval_context_relevance", "context_tokens",
                        )} | {"response": {"status": response.status}})
                        write_json(output / "summary.json", summarize(rows))
                        progress.update(1)
    except BaseException:
        stopped = True
        raise
    finally:
        manifest["status"] = "interrupted" if stopped else "completed"
        try:
            after = snapshot() if snapshot else None
            manifest["snapshot_after"] = after
            manifest["snapshot_consistent"] = before == after if snapshot else None
        except Exception as error:
            manifest["snapshot_error"] = exception_diagnostics(error)
            manifest["snapshot_consistent"] = False
        summary = summarize(rows)
        summary["snapshot_consistent"] = manifest["snapshot_consistent"]
        summary["expected_rows"] = len(cases) * len(settings.conditions) * 2
        write_json(output / "summary.json", summary)
        write_json(output / "manifest.json", manifest)
        try:
            from graphrag_pipeline.evaluation.report import generate_report
            report = generate_report(output)
            summary["report"] = report["report"]
            summary["report_warnings"] = report["warnings"]
        except Exception as error:
            diagnostics = exception_diagnostics(error)
            logging.getLogger(__name__).error("HTML report generation failed; evaluation artifacts are preserved: %s", diagnostics)
            # An interruption or original evaluation failure must retain its cause.
            if not stopped:
                raise RuntimeError(f"HTML report generation failed; evaluation artifacts remain in {output}: {diagnostics}") from error
    return rows, summary
