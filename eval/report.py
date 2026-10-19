"""Evaluation report generator with paired bootstrap confidence intervals.

Standard evaluation report format for all GraphRAG retrieval and QA experiments.
Computes Recall@2, Recall@5, Later-hop Recall@5, Support F1, Answer F1/EM,
cost per question, latency p50/p95, and 95% paired bootstrap confidence intervals.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def paired_bootstrap_ci(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    num_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Compute 95% paired bootstrap confidence interval for difference (mean(a) - mean(b)).

    Args:
        a: Metric values for system A across N questions.
        b: Metric values for system B across the same N questions.
        num_bootstrap: Number of bootstrap iterations (default 1000).
        alpha: Significance level (default 0.05 for 95% CI).
        seed: Optional random seed for deterministic reproducibility.

    Returns:
        tuple (observed_difference, ci_lower_bound, ci_upper_bound)
    """
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    if len(arr_a) != len(arr_b):
        raise ValueError(f"Paired bootstrap requires equal lengths: {len(arr_a)} vs {len(arr_b)}")
    if len(arr_a) == 0:
        return 0.0, 0.0, 0.0

    diffs = arr_a - arr_b
    observed_diff = float(np.mean(diffs))

    n = len(diffs)
    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, n, size=(num_bootstrap, n))
    boot_means = np.mean(diffs[boot_indices], axis=1)

    low = float(np.percentile(boot_means, 100 * (alpha / 2)))
    high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return observed_diff, low, high


@dataclass
class SystemMetrics:
    """Metrics for one evaluated system/run."""

    system_name: str
    setting: str  # "per_question" or "pooled"
    split: str = "test"  # "tune", "test", "all"
    num_questions: int = 0
    recall_at_2: float = 0.0
    recall_at_5: float = 0.0
    later_hop_recall_at_5: float = 0.0
    support_f1: float = 0.0
    answer_f1: float = 0.0
    answer_em: float = 0.0
    cost_per_question: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    per_question_scores: dict[str, list[float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Exclude raw arrays from summary dict for clean JSON
        d.pop("per_question_scores", None)
        return d


def compare_systems(
    system_a: SystemMetrics,
    system_b: SystemMetrics,
    metric: str = "recall_at_5",
    num_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute paired bootstrap difference between system A and system B on a given metric."""
    scores_a = system_a.per_question_scores.get(metric, [])
    scores_b = system_b.per_question_scores.get(metric, [])

    if not scores_a or not scores_b:
        # Fall back to scalar difference if per-question scores are missing
        val_a = getattr(system_a, metric, 0.0)
        val_b = getattr(system_b, metric, 0.0)
        diff = val_a - val_b
        return {
            "metric": metric,
            "system_a": system_a.system_name,
            "system_b": system_b.system_name,
            "observed_diff": round(diff, 2),
            "ci_lower": round(diff, 2),
            "ci_upper": round(diff, 2),
            "is_paired": False,
        }

    diff, low, high = paired_bootstrap_ci(
        scores_a, scores_b, num_bootstrap=num_bootstrap, seed=seed
    )
    return {
        "metric": metric,
        "system_a": system_a.system_name,
        "system_b": system_b.system_name,
        "observed_diff": round(diff, 2),
        "ci_lower": round(low, 2),
        "ci_upper": round(high, 2),
        "is_paired": True,
    }


def format_markdown_table(systems: list[SystemMetrics]) -> str:
    """Format evaluation systems into a standard GitHub Markdown table."""
    headers = [
        "System",
        "Setting",
        "Split",
        "R@2 (%)",
        "R@5 (%)",
        "Later-hop R@5 (%)",
        "Support F1 (%)",
        "Answer F1 (%)",
        "Answer EM (%)",
        "Cost/q ($)",
        "Latency p50 (ms)",
        "Latency p95 (ms)",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for s in systems:
        row = [
            s.system_name,
            s.setting,
            s.split,
            f"{s.recall_at_2:.2f}",
            f"{s.recall_at_5:.2f}",
            f"{s.later_hop_recall_at_5:.2f}",
            f"{s.support_f1:.2f}",
            f"{s.answer_f1:.2f}",
            f"{s.answer_em:.2f}",
            f"${s.cost_per_question:.4f}",
            f"{s.latency_p50_ms:.1f}",
            f"{s.latency_p95_ms:.1f}",
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def parse_baseline_output_to_systems(
    baseline_json: dict[str, Any],
) -> list[SystemMetrics]:
    """Parse baseline results JSON into SystemMetrics rows."""
    systems: list[SystemMetrics] = []

    for split in ("tune", "test"):
        if split not in baseline_json:
            continue
        split_data = baseline_json[split]

        # Per-question BM25
        if "per_question_bm25" in split_data:
            b = split_data["per_question_bm25"]
            systems.append(
                SystemMetrics(
                    system_name="BM25",
                    setting="per_question",
                    split=split,
                    num_questions=b.get("question_count", 0),
                    recall_at_2=b.get("recall_at_2", 0.0),
                    recall_at_5=b.get("recall_at_5", 0.0),
                    later_hop_recall_at_5=b.get("later_hop_recall_at_5", 0.0),
                    support_f1=b.get("support_f1", 0.0),
                    cost_per_question=0.0,
                    per_question_scores=b.get("per_question_scores", {}),
                )
            )

        # Pooled BM25
        if "pooled_bm25" in split_data:
            p = split_data["pooled_bm25"]
            systems.append(
                SystemMetrics(
                    system_name="BM25",
                    setting="pooled",
                    split=split,
                    num_questions=p.get("question_count", 0),
                    recall_at_2=p.get("recall_at_2", 0.0),
                    recall_at_5=p.get("recall_at_5", 0.0),
                    later_hop_recall_at_5=p.get("later_hop_recall_at_5", 0.0),
                    support_f1=p.get("support_f1", 0.0),
                    cost_per_question=0.0,
                    per_question_scores=p.get("per_question_scores", {}),
                )
            )

        # Random-5 floor
        if "random_5" in split_data:
            r = split_data["random_5"]
            systems.append(
                SystemMetrics(
                    system_name="Random-5 Floor",
                    setting="per_question",
                    split=split,
                    num_questions=r.get("question_count", 0),
                    recall_at_2=0.0,
                    recall_at_5=r.get("recall_at_k", 0.0),
                    later_hop_recall_at_5=r.get("later_hop_recall_at_5", 0.0),
                    support_f1=0.0,
                    cost_per_question=0.0,
                    per_question_scores=r.get("per_question_scores", {}),
                )
            )

    return systems


def build_full_report_markdown(
    report_title: str,
    systems: list[SystemMetrics],
    comparisons: list[dict[str, Any]] | None = None,
    notes: list[str] | None = None,
) -> str:
    """Build a complete formatted markdown document for an evaluation report."""
    md_parts = [
        f"# {report_title}",
        "",
        "## Performance Summary",
        "",
        format_markdown_table(systems),
        "",
    ]

    if comparisons:
        md_parts.extend([
            "## Paired Comparisons (95% Bootstrap Confidence Intervals)",
            "",
            "| System A | System B | Metric | Difference (A - B) | 95% Confidence Interval |",
            "| --- | --- | --- | --- | --- |",
        ])
        for c in comparisons:
            diff_str = f"{c['observed_diff']:+.2f}"
            ci_str = f"[{c['ci_lower']:+.2f}, {c['ci_upper']:+.2f}]"
            md_parts.append(
                f"| {c['system_a']} | {c['system_b']} | {c['metric']} | {diff_str} | {ci_str} |"
            )
        md_parts.append("")

    if notes:
        md_parts.extend(["## Evaluation Notes", ""])
        for note in notes:
            md_parts.append(f"- {note}")
        md_parts.append("")

    return "\n".join(md_parts)
