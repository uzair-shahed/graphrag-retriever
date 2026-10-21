"""Offline comparisons of validated benchmark artifacts; HTML and SVG, no services."""

from __future__ import annotations

import base64
import json
import math
import textwrap
from html import escape
from pathlib import Path

from graphrag_pipeline.extraction_benchmark.artifacts import (
    APPEND_TRANSACTION, atomic_text, load_gold, load_run, read_json,
)
from graphrag_pipeline.extraction_benchmark.runner import summarize

GENERATOR = "graphrag-extraction-benchmark.visualize"
FILES = ("entities.svg", "relations.svg", "outcomes.svg", "reasoning-delta.svg", "index.html", "comparison.json")
COLORS = {"precision": "#2463a6", "recall": "#007d75", "f1": "#824bb0",
          "succeeded": "#007d75", "failed": "#c24b36", "pending": "#718096"}


def _runs(values):
    parsed, labels, paths = [], set(), set()
    for value in values:
        label, separator, raw_path = value.partition("=")
        label, raw_path = label.strip(), raw_path.strip()
        if not separator or not label or not raw_path:
            raise ValueError("--run must be LABEL=PATH, for example --run Qwen-off=runs/entity-extraction/qwen-off")
        path = Path(raw_path).resolve()
        if label in labels or path in paths:
            raise ValueError(f"duplicate run label or directory: {value}; specify each directory once")
        labels.add(label)
        paths.add(path)
        parsed.append((label, path))
    if not parsed:
        raise ValueError("at least one --run LABEL=PATH is required")
    return parsed


def _inactive(directory):
    if not directory.is_dir():
        raise ValueError(f"benchmark directory does not exist: {directory}")
    if (directory / ".benchmark.lock").exists() or (directory / APPEND_TRANSACTION).exists():
        raise ValueError(f"finish/resume the benchmark writer before visualizing: {directory}")


def _reasoning(settings):
    value = settings.get("think" if settings["provider"] == "ollama" else "reasoning_effort")
    if value is None:
        return "unspecified", "unspecified (provider default)"
    if value is False or value == "none":
        return "off", "off"
    if value is True:
        return "on", "on"
    if isinstance(value, str) and value.strip():
        return "on", f"on ({value})"
    raise ValueError(f"invalid reasoning setting for {settings['provider']}:{settings['model']}")


def _comparison(gold, runs):
    _inactive(gold)
    requests, reference, fingerprint, skipped = load_gold(gold)
    gold_manifest = read_json(gold / "manifest.json")
    series, warnings = [], []
    if skipped:
        warnings.append(f"Gold has {len(skipped)} failed/missing labels. These paragraphs are excluded for every candidate.")
    if not reference:
        raise ValueError("gold has no usable labels; finish/retry gold generation before visualizing")
    for label, path in runs:
        _inactive(path)
        try:
            manifest, cohort, rows, frozen_reference = load_run(path)
        except ValueError as failure:
            raise ValueError(f"{label} ({path}): {failure}") from failure
        config = manifest["configuration"]
        if manifest["kind"] != "candidate":
            raise ValueError(f"{label}: --run must point to a candidate run, not gold")
        if cohort != requests:
            raise ValueError(f"{label}: cohort differs from --gold; select runs using exactly the same paragraphs")
        if config.get("gold_fingerprint") != fingerprint:
            raise ValueError(f"{label}: gold fingerprint differs from --gold; select the original matching gold directory")
        if frozen_reference != reference or config.get("skipped_gold", []) != skipped:
            raise ValueError(f"{label}: frozen reference labels/skips differ from --gold; use the original matching gold")
        summary = summarize(manifest, cohort, rows)
        for settings in config["models"]:
            model = f"{settings['provider']}:{settings['model']}"
            state, reasoning_label = _reasoning(settings)
            stats = summary["models"][model]
            item = {"label": label, "model": model, "title": f"{label} | {model} | reasoning {reasoning_label}",
                    "run_id": manifest["run_id"], "directory": str(path), "reasoning": state,
                    "reasoning_label": reasoning_label, "settings": settings,
                    "prompt": config["prompt"], "response_schema": config["response_schema"],
                    **stats}
            series.append(item)
            if stats["pending"]:
                warnings.append(f"{item['title']}: {stats['pending']} pending paragraphs; metrics cover only attempted rows.")
            if stats["generation_formats"]["legacy_v1"]:
                warnings.append(f"{item['title']}: {stats['generation_formats']['legacy_v1']} saved generations use the legacy response format.")
    # Report every pair explicitly. Repeated experiments are never silently averaged.
    deltas = []
    for on in series:
        for off in series:
            if (on["model"] != off["model"] or on["reasoning"] != "on" or off["reasoning"] != "off"
                    or on["pending"] or off["pending"]):
                continue
            changed = [key for key in sorted(set(on["settings"]) | set(off["settings"]))
                       if key not in {"reasoning_effort", "think"}
                       and on["settings"].get(key) != off["settings"].get(key)]
            changed += [key for key in ("prompt", "response_schema", "generation_formats") if on[key] != off[key]]
            deltas.append({"model": on["model"], "on": on["title"], "off": off["title"],
                           "changed_settings": changed,
                           "entities": 100 * (on["micro"]["entities"]["f1"] - off["micro"]["entities"]["f1"]),
                           "relations": 100 * (on["micro"]["relations"]["f1"] - off["micro"]["relations"]["f1"])})
    for _, path in runs:
        _inactive(path)
    _inactive(gold)
    return {"generator": GENERATOR, "format_version": 1, "generated_files": list(FILES),
            "gold": {"directory": str(gold), "run_id": gold_manifest["run_id"],
                     "model": gold_manifest["configuration"]["models"][0], "fingerprint": fingerprint,
                     "cohort_sha256": gold_manifest["configuration"]["cohort_sha256"],
                     "requested": len(requests), "comparable": len(reference), "skipped": len(skipped)},
            "scoring_version": gold_manifest["configuration"]["scoring_version"],
            "series": series, "reasoning_deltas": deltas, "warnings": warnings}


def _text(x, y, value, **attrs):
    attributes = " ".join(f'{key.replace("_", "-")}="{escape(str(val), quote=True)}"' for key, val in attrs.items())
    return f'<text x="{x}" y="{y}" {attributes}>{escape(str(value))}</text>'


def _wrapped(parts, value, y):
    for line in textwrap.wrap(value, width=105, break_long_words=True) or [""]:
        parts.append(_text(20, y, line, font_weight="600"))
        y += 19
    return y + 5


def _svg(title, parts, height):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}" '
            f'viewBox="0 0 1000 {height}" role="img">'
            f'<title>{escape(title)}</title><rect width="100%" height="100%" fill="white"/>'
            '<g font-family="Arial, sans-serif" font-size="14" fill="#183047">'
            + _text(20, 30, title, font_size="21", font_weight="700")
            + "".join(parts) + '</g></svg>')


def _quality_chart(data, task):
    parts, y = [], 65
    for item in data["series"]:
        y = _wrapped(parts, item["title"], y)
        for metric in ("precision", "recall", "f1"):
            value = item["micro"][task][metric]
            parts += [_text(20, y + 13, metric.capitalize()),
                      f'<rect x="120" y="{y}" width="740" height="17" fill="#eef2f6"/>',
                      f'<rect x="120" y="{y}" width="{740 * value:.4f}" height="17" fill="{COLORS[metric]}"/>',
                      _text(875, y + 13, f"{100 * value:.2f}%")]
            y += 25
        y += 18
    parts.append(_text(20, y, "Scale: 0-100%. Micro metrics sum TP/FP/FN across attempted paragraphs."))
    return _svg(f"{task.capitalize()}: precision, recall and F1", parts, y + 25)


def _outcome_chart(data):
    parts, y = [], 65
    for item in data["series"]:
        y = _wrapped(parts, item["title"], y)
        x = 20
        for status in ("succeeded", "failed", "pending"):
            width = 870 * item[status] / item["expected"]
            parts.append(f'<rect x="{x:.4f}" y="{y}" width="{width:.4f}" height="22" fill="{COLORS[status]}"/>')
            x += width
        y += 43
        parts.append(_text(20, y, f"Succeeded: {item['succeeded']} | Failed: {item['failed']} | Pending: {item['pending']} | Eligible: {item['expected']}"))
        y += 34
    parts.append(_text(20, y, "Green = succeeded; red = failed; gray = pending. Outcomes count paragraph extractions."))
    return _svg("Extraction outcomes", parts, y + 25)


def _delta_chart(data):
    parts, y = [], 65
    deltas = data["reasoning_deltas"]
    if not deltas:
        parts.append(_text(20, y, "No completed on/off pair is available for the same provider and model."))
        return _svg("Reasoning: F1 change (on minus off)", parts, y + 30)
    maximum = max(abs(pair[task]) for pair in deltas for task in ("entities", "relations"))
    limit = max(5, math.ceil(maximum / 5) * 5)
    for pair in deltas:
        y = _wrapped(parts, f"ON: {pair['on']}", y)
        y = _wrapped(parts, f"OFF: {pair['off']}", y)
        for task in ("entities", "relations"):
            value = pair[task]
            width = 335 * abs(value) / limit
            x = 490 if value >= 0 else 490 - width
            parts += [_text(20, y + 14, task.capitalize()),
                      f'<rect x="155" y="{y}" width="670" height="18" fill="#eef2f6"/>',
                      f'<rect x="{x:.4f}" y="{y}" width="{width:.4f}" height="18" fill="{COLORS["f1"]}"/>',
                      f'<line x1="490" y1="{y - 2}" x2="490" y2="{y + 21}" stroke="#183047"/>',
                      _text(840, y + 14, f"{value:+.2f} pp")]
            y += 28
        y += 15
    parts.append(_text(20, y, f"Symmetric scale: -{limit} to +{limit} percentage points. Positive values favor reasoning on."))
    return _svg("Reasoning: F1 change (on minus off)", parts, y + 25)


def _table(headers, rows):
    return ('<div class="scroll"><table><thead><tr>'
            + ''.join(f'<th scope="col">{escape(header)}</th>' for header in headers)
            + '</tr></thead><tbody>'
            + ''.join('<tr>' + ''.join(f'<td>{escape(str(cell))}</td>' for cell in row) + '</tr>' for row in rows)
            + '</tbody></table></div>')


def _html(data, charts):
    gold = data["gold"]
    model = gold["model"]
    parts = [f'<p>Gold: {escape(model["provider"] + ":" + model["model"])}. '
             f'{gold["requested"]} paragraphs; {gold["comparable"]} usable gold labels; {gold["skipped"]} skipped.</p>',
             '<p>Entity matching compares normalized mention text and entity type. Relation matching compares directed endpoints and relationship type. '
             'Scores use normalized exact matching. Failed extractions count as empty predictions; pending paragraphs and unusable gold are excluded.</p>']
    for warning in data["warnings"]:
        parts.append(f'<p class="notice">{escape(warning)}</p>')
    for filename, title in (("entities.svg", "Entity quality"), ("relations.svg", "Relation quality"),
                            ("outcomes.svg", "Extraction outcomes"), ("reasoning-delta.svg", "Reasoning comparison")):
        encoded = base64.b64encode(charts[filename].encode("utf-8")).decode("ascii")
        parts.append(f'<section><h2>{title}</h2><a download="{filename}" href="data:image/svg+xml;base64,{encoded}">Download SVG</a>'
                     + charts[filename] + '</section>')
    parts.append('<section><h2>Exact metrics</h2>')
    rows = []
    for item in data["series"]:
        for task in ("entities", "relations"):
            metrics = item["micro"][task]
            rows.append([item["label"], item["model"], item["reasoning_label"], task,
                         *(f'{100 * metrics[key]:.2f}%' for key in ("precision", "recall", "f1")),
                         *(metrics[key] for key in ("tp", "fp", "fn")),
                         item["succeeded"], item["failed"], item["pending"]])
    parts.append(_table(["Run", "Model", "Reasoning", "Task", "Precision", "Recall", "F1", "TP", "FP", "FN",
                         "Succeeded", "Failed", "Pending"], rows))
    parts.append('<p>Succeeded/failed/pending counts describe the same paragraph extractions for both tasks.</p></section>')
    parts.append('<section><h2>Interpreting reasoning differences</h2><p>Each completed on/off run pair for the same provider/model is shown separately. '
                 'Unspecified settings mean provider default and are not classified as on or off. '
                 'Differences describe these saved experiments; they do not establish statistical significance or isolate reasoning when other settings differ.</p>')
    if data["reasoning_deltas"]:
        parts.append(_table(["Reasoning on", "Reasoning off", "Entity F1 change", "Relation F1 change", "Other differences"],
                     [[pair["on"], pair["off"], f'{pair["entities"]:+.2f} pp', f'{pair["relations"]:+.2f} pp',
                       ", ".join(pair["changed_settings"]) or "None recorded"] for pair in data["reasoning_deltas"]]))
    parts.append('</section><section><h2>Run configuration and provenance</h2>')
    parts.append(_table(["Run", "Model", "Run ID", "Token budget", "Timeout", "Retries", "Legacy / v2 generations"],
                 [[item["label"], item["model"], item["run_id"], item["settings"].get("max_tokens"),
                   item["settings"].get("timeout"), item["settings"].get("retries"),
                   f'{item["generation_formats"]["legacy_v1"]} / {item["generation_formats"]["simplified_v2"]}']
                  for item in data["series"]]))
    parts.append(f'<p>Scoring: {escape(data["scoring_version"])}.</p><p>Gold fingerprint: <code>{escape(gold["fingerprint"])}</code></p>'
                 f'<p>Cohort hash: <code>{escape(gold["cohort_sha256"])}</code></p></section>')
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1"><title>Extraction benchmark comparison</title>'
            '<style>body{font:16px/1.55 system-ui,sans-serif;color:#183047;background:#f3f6f9;margin:0}'
            'main{max-width:1200px;margin:auto;padding:28px}section{background:white;padding:22px;margin:22px 0;border-radius:10px}'
            'svg{display:block;width:100%;height:auto;margin-top:18px}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}'
            'th,td{padding:10px;text-align:left;border-bottom:1px solid #dbe3eb}th{background:#eef2f6}'
            'a{color:#2463a6}.notice{padding:12px;border-left:4px solid #bd7b16;background:#fff5da}code{overflow-wrap:anywhere}'
            '@media print{body{background:white}section{break-inside:avoid}a{display:none}main{padding:0}}</style></head>'
            '<body><main><h1>Extraction benchmark comparison</h1>' + ''.join(parts) + '</main></body></html>')


def visualize(gold: Path, runs: list[str], output: Path):
    """Validate inputs, recompute aggregates and atomically write only owned report files."""
    gold, output = Path(gold).resolve(), Path(output).resolve()
    parsed = _runs(runs)
    if output.exists() and not output.is_dir():
        raise ValueError("visualization --output must be a directory")
    for source in [gold, *(path for _, path in parsed)]:
        if source == output or source in output.parents or output in source.parents:
            raise ValueError("visualization output must be separate from all input directories, without nesting")
    data = _comparison(gold, parsed)
    charts = {"entities.svg": _quality_chart(data, "entities"), "relations.svg": _quality_chart(data, "relations"),
              "outcomes.svg": _outcome_chart(data), "reasoning-delta.svg": _delta_chart(data)}
    contents = {**charts, "index.html": _html(data, charts),
                "comparison.json": json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"}
    owned = set()
    marker = output / "comparison.json"
    if marker.is_file() and not marker.is_symlink():
        try:
            previous = read_json(marker)
            if previous.get("generator") == GENERATOR and previous.get("format_version") == 1:
                owned = set(previous.get("generated_files", [])) & set(FILES)
        except (ValueError, AttributeError, TypeError):
            pass
    for name in FILES:
        path = output / name
        if path.is_symlink() or (path.exists() and (name not in owned or not path.is_file())):
            raise ValueError(f"output file is not owned by the visualizer: {path}; choose another --output directory")
    output.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        atomic_text(output / name, contents[name])
    return {"output": str(output), "report": str(output / "index.html"), "models_and_runs": len(data["series"]),
            "reasoning_pairs": len(data["reasoning_deltas"]), "warnings": data["warnings"], "provider_calls": 0}
