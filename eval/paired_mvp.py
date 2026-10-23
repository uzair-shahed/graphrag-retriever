"""Paired question-by-question comparison of a saved MVP run.

Uses the official per-question answer scorer and the same support-F1 definition
as eval/run_mvp.py. Recall@K is not recomputed here: the saved prediction files
do not store the retrieved paragraph sets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

_third_party_musique = _project_root / "third_party" / "musique"
if _third_party_musique.exists() and str(_third_party_musique) not in sys.path:
    sys.path.insert(0, str(_third_party_musique))

from eval.run_mvp import bootstrap_ci
from metrics.answer import compute_exact, compute_f1, metric_max_over_ground_truths


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def index_by_id(rows: list[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = str(row.get("id", ""))
        if not row_id:
            raise ValueError(f"{label} has a row without an id")
        if row_id in indexed:
            raise ValueError(f"{label} repeats id {row_id}")
        indexed[row_id] = row
    return indexed


def question_scores(prediction: dict[str, Any], gold: dict[str, Any]) -> dict[str, float]:
    """Score one question the same way the MVP runner builds its intervals."""
    gold_answers = [gold["answer"]] + list(gold.get("answer_aliases") or [])
    predicted = prediction.get("predicted_answer", "")
    scores = {
        "answer_f1": float(metric_max_over_ground_truths(compute_f1, predicted, gold_answers)),
        "answer_em": float(metric_max_over_ground_truths(compute_exact, predicted, gold_answers)),
    }
    predicted_lower = str(predicted).strip().lower()
    contained = any(
        str(answer).strip().lower() in predicted_lower
        for answer in gold_answers
        if str(answer).strip()
    )
    scores["answer_containment"] = 1.0 if contained else 0.0

    gold_support = {int(paragraph["idx"]) for paragraph in gold.get("paragraphs", []) if paragraph.get("is_supporting")}
    predicted_support = {int(idx) for idx in prediction.get("predicted_support_idxs", [])}
    true_positives = len(predicted_support & gold_support)
    precision = true_positives / len(predicted_support) if predicted_support else 0.0
    recall = true_positives / len(gold_support) if gold_support else 0.0
    if not predicted_support and not gold_support:
        support_f1 = 1.0
    elif precision + recall == 0:
        support_f1 = 0.0
    else:
        support_f1 = (2 * precision * recall) / (precision + recall)
    scores["support_f1"] = float(support_f1)
    return scores


def paired_difference(
    left: dict[str, dict[str, float]],
    right: dict[str, dict[str, float]],
    question_ids: list[str],
    metric: str,
) -> dict[str, Any]:
    """Mean of left minus right on the same questions, with a bootstrap interval."""
    differences = [left[qid][metric] - right[qid][metric] for qid in question_ids]
    higher = sum(1 for value in differences if value > 0)
    lower = sum(1 for value in differences if value < 0)
    ties = len(differences) - higher - lower
    mean = sum(differences) / len(differences) if differences else 0.0
    low, high = bootstrap_ci(differences) if differences else (0.0, 0.0)
    return {
        "metric": metric,
        "n": len(differences),
        "mean": round(mean, 6),
        "ci_low": low,
        "ci_high": high,
        "higher": higher,
        "tie": ties,
        "lower": lower,
        "separates": low > 0 or high < 0,
    }


def compare_runs(
    gold_rows: list[dict[str, Any]],
    predictions_by_condition: dict[str, list[dict[str, Any]]],
    pairs: list[tuple[str, str]],
) -> dict[str, Any]:
    gold = index_by_id(gold_rows, label="gold")
    question_ids = list(gold)
    scored: dict[str, dict[str, dict[str, float]]] = {}
    for condition, rows in predictions_by_condition.items():
        indexed = index_by_id(rows, label=condition)
        missing = [qid for qid in question_ids if qid not in indexed]
        extra = [qid for qid in indexed if qid not in gold]
        if missing or extra:
            raise ValueError(f"{condition} does not match gold ids: missing {len(missing)}, extra {len(extra)}")
        scored[condition] = {qid: question_scores(indexed[qid], gold[qid]) for qid in question_ids}

    comparisons = []
    for left_name, right_name in pairs:
        metrics = []
        for metric in ("answer_f1", "answer_em", "answer_containment", "support_f1"):
            metrics.append(paired_difference(scored[left_name], scored[right_name], question_ids, metric))
        comparisons.append({"left": left_name, "right": right_name, "metrics": metrics})
    return {"n_questions": len(question_ids), "comparisons": comparisons}


def render_paired_markdown(report: dict[str, Any]) -> str:
    lines = [
        "### Paired differences",
        "",
        f"Each number is the mean of {report['n_questions']} question-level differences, left minus right. The interval is a 95% bootstrap of that mean, using the same seed as the MVP runner. An interval that stays above 0 separates the systems in favor of the left condition. An interval that contains 0 does not.",
        "",
        "Recall@2, Recall@5, and later-hop Recall@5 are absent because the saved prediction files do not store the retrieved paragraph sets.",
        "",
    ]
    for comparison in report["comparisons"]:
        lines.extend([
            f"#### `{comparison['left']}` minus `{comparison['right']}`",
            "",
            "| Metric | Mean difference | 95% CI | Left higher | Tie | Right higher |",
            "| :--- | ---: | :--- | ---: | ---: | ---: |",
        ])
        for metric in comparison["metrics"]:
            lines.append(
                "| {metric} | {mean:+.3f} | [{low:+.3f}, {high:+.3f}] | {higher} | {tie} | {lower} |".format(
                    metric=metric["metric"],
                    mean=metric["mean"],
                    low=metric["ci_low"],
                    high=metric["ci_high"],
                    higher=metric["higher"],
                    tie=metric["tie"],
                    lower=metric["lower"],
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired comparison of a saved MVP prediction directory.")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/mvp_results"))
    parser.add_argument("--gold", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=Path("eval/reports/mvp_test300_paired.json"))
    parser.add_argument("--output-md", type=Path, default=Path("eval/reports/mvp_test300_paired.md"))
    args = parser.parse_args()

    gold_path = args.gold or (args.run_dir / "gold_eval_records.jsonl")
    conditions = ("bm25", "vector", "walk_off", "walk_exact")
    predictions = {
        condition: load_jsonl(args.run_dir / f"predictions_{condition}.jsonl")
        for condition in conditions
    }
    report = compare_runs(
        load_jsonl(gold_path),
        predictions,
        pairs=[
            ("walk_exact", "vector"),
            ("walk_exact", "bm25"),
            ("vector", "bm25"),
            ("walk_off", "vector"),
        ],
    )
    markdown = render_paired_markdown(report)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
