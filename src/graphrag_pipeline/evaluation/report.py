"""Offline, artifact-only HTML reports. No model or graph clients are used here."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

from graphrag_pipeline.answer_context import render_evidence_context
from graphrag_pipeline.evaluation.records import GoldRecord
from graphrag_pipeline.query import EvidenceChunk, QueryResponse

BACKENDS = ("vectorrag", "graphrag")
METRICS = ("faithfulness", "answer_relevancy", "context_relevance")


def _json(text):
    def reject(value):
        raise ValueError(f"non-finite JSON number: {value}")
    return json.loads(text, parse_constant=reject)


def _embedded(value):
    """JSON inside a script element must not contain an HTML closing tag."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace(
        "&", "\\u0026").replace("<", "\\u003c").replace(
        ">", "\\u003e").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def _validate_row(row, manifest):
    gold = GoldRecord.model_validate(row["labels"])
    response = QueryResponse.model_validate(row["response"])
    if (row["run_id"] != manifest["run_id"] or row["record_id"] != gold.id
            or response.question != gold.question or row["request"]["question"] != gold.question
            or row["trace_id"] != response.trace_id or response.retrieval_backend != row["backend"]):
        raise ValueError("row identity, question, or gold labels disagree")
    if row["condition"] not in manifest["settings"]["conditions"] or row["backend"] not in BACKENDS:
        raise ValueError("unknown condition or backend")
    full = [EvidenceChunk.model_validate(e) for e in row["retrieval_evidence"]]
    shown = [EvidenceChunk.model_validate(e) for e in row["answer_evidence"]]
    if response.evidence != shown or any(e not in full for e in shown):
        raise ValueError("retrieved, answer, and response evidence disagree")
    if not set(response.citations) <= {e.chunk_id for e in shown}:
        raise ValueError("citation outside answer evidence")
    if not isinstance(row["answer_context"], str):
        raise ValueError("answer_context must be text")
    if row["answer_context"] != render_evidence_context(shown):
        raise ValueError("answer_context disagrees with answer evidence")
    prompt = row["answer_prompt"]
    if prompt is not None and (not isinstance(prompt, str) or row["answer_context"] not in prompt):
        raise ValueError("answer prompt disagrees with answer context")
    for score in [*(row["ragas"][m] for m in METRICS), row["retrieval_context_relevance"]]:
        if score["status"] not in ("scored", "failed", "not_applicable"):
            raise ValueError("unknown metric status")
        value = score["value"]
        if score["status"] == "scored":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("scored metric requires a finite number")
        elif value is not None:
            raise ValueError("unscored metric must have a null value")
    # Aggregation below also checks the required gold, timing and coverage fields.
    return gold.model_dump(mode="json")


def load_report_data(directory):
    """Validate saved rows, deduplicate per record, and recompute faithful aggregates."""
    from graphrag_pipeline.evaluation.runner import summarize

    directory = Path(directory)
    manifest = _json((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported evaluation manifest schema_version")
    conditions = manifest["settings"]["conditions"]
    if not conditions or any(c not in ("record", "corpus") for c in conditions) or len(set(conditions)) != len(conditions):
        raise ValueError("invalid manifest conditions")
    selected = manifest["selected_records"]
    records = {r["id"]: {"id": r["id"], "source_line": r["source_line"], "labels": None,
                           "evidence": [], "texts": [], "results": {}} for r in selected}
    if not records or len(records) != len(selected):
        raise ValueError("invalid or duplicate selected records")
    # Drop canonical lookup strings after interning; the retained data contains
    # just one copy of each evidence payload and context per selected record.
    pools = {rid: ({}, {}) for rid in records}
    warnings, aggregates, traces = [], [], set()
    with (directory / "records.jsonl").open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = _json(line)
            except json.JSONDecodeError as error:
                remainder = line[error.pos:].strip()
                incomplete = (error.pos >= len(line.rstrip())
                              or error.msg.startswith("Unterminated string")
                              or bool(remainder) and any(word.startswith(remainder) for word in ("true", "false", "null")))
                if not line.endswith("\n") and incomplete:
                    warnings.append(f"Ignored incomplete final JSONL line {line_number}; only saved complete rows are shown.")
                    break
                raise ValueError(f"records.jsonl:{line_number}: malformed complete row") from error
            try:
                labels = _validate_row(row, manifest)
                record = records[row["record_id"]]
                key = f'{row["condition"]}/{row["backend"]}'
                if key in record["results"] or row["trace_id"] in traces:
                    raise ValueError("duplicate result or trace ID")
                if record["labels"] is not None and record["labels"] != labels:
                    raise ValueError("inconsistent gold labels for record")
                record["labels"] = labels
                traces.add(row["trace_id"])
                aggregates.append({k: row[k] for k in (
                    "pair_id", "condition", "backend", "ragas", "gold", "retrieval_gold",
                    "graph_component_complete", "paired_seeds_consistent", "timings",
                    "answer_context_gold", "retrieval_context_relevance", "context_tokens",
                )} | {"response": {"status": row["response"]["status"]}})
                evidence_pool, text_pool = pools[row["record_id"]]

                def intern(value, lookup, values):
                    canonical = _embedded(value)
                    if canonical not in lookup:
                        lookup[canonical] = len(values)
                        values.append(value)
                    return lookup[canonical]

                compact = {k: v for k, v in row.items() if k not in (
                    "labels", "response", "retrieval_evidence", "answer_evidence", "answer_context", "answer_prompt")}
                compact["response"] = {k: v for k, v in row["response"].items() if k != "evidence"}
                for field in ("retrieval_evidence", "answer_evidence"):
                    compact[field] = [intern(e, evidence_pool, record["evidence"]) for e in row[field]]
                for field in ("answer_context", "answer_prompt"):
                    compact[field] = intern(row.get(field), text_pool, record["texts"])
                record["results"][key] = compact
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"records.jsonl:{line_number}: {error}") from error
    try:
        summary = summarize(aggregates)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid saved aggregate fields: {error}") from error
    expected = len(records) * len(conditions) * 2
    summary.update(expected_rows=expected, snapshot_consistent=manifest.get("snapshot_consistent"))
    saved_summary = directory / "summary.json"
    if saved_summary.exists():
        saved = _json(saved_summary.read_text(encoding="utf-8"))
        if saved != summary:
            warnings.append("Saved summary differs from complete rows; charts were recomputed from those rows.")
    else:
        warnings.append("summary.json is absent; charts were recomputed from saved rows.")
    if len(aggregates) != expected or manifest.get("status") != "completed":
        warnings.append(f"Partial run: {len(aggregates)} of {expected} expected results; missing results are marked below.")
    if manifest.get("snapshot_consistent") is not True:
        warnings.append("Graph snapshot consistency was not verified; interpret comparisons with care.")
    return {"manifest": manifest, "summary": summary, "warnings": warnings, "records": list(records.values())}


def generate_report(directory):
    """Atomically replace only report.html; return its path and visible warnings."""
    directory = Path(directory)
    data = load_report_data(directory)
    records = data.pop("records")
    data["records"] = [{"id": r["id"], "question": r["labels"]["question"] if r["labels"] else "No saved result"} for r in records]
    template = Path(__file__).with_name("report_template.html").read_text(encoding="utf-8")
    prefix, suffix = template.replace("__REPORT_HEADER__", _embedded(data)).split("__REPORT_RECORDS__")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, suffix=".html.tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(prefix)
            for index, record in enumerate(records):
                stream.write(f'<script type="application/json" id="record-data-{index}">{_embedded(record)}</script>\n')
            stream.write(suffix)
        destination = directory / "report.html"
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"report": str(destination), "warnings": data["warnings"], "completed_rows": data["summary"]["completed_rows"]}
