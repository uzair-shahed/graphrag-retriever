"""Unit tests for evaluation report generator and paired bootstrap confidence intervals."""

import numpy as np
import pytest

from eval.report import (
    SystemMetrics,
    build_full_report_markdown,
    compare_systems,
    format_markdown_table,
    paired_bootstrap_ci,
    parse_baseline_output_to_systems,
)


def test_paired_bootstrap_ci_coverage_on_synthetic_trials():
    """Verify paired bootstrap 95% CI on synthetic predictions with known differences.

    Pass criterion from implementation plan M2.4:
    The 95% interval contains the true difference in >= 90 of 100 seeded trials.
    """
    true_difference = 4.5
    n_questions = 200
    trials = 100
    contained_count = 0

    for trial_idx in range(trials):
        rng = np.random.default_rng(trial_idx)
        # Shared question difficulty baseline
        base_scores = rng.uniform(20.0, 70.0, size=n_questions)
        # Add correlated noise and known treatment difference
        noise_a = rng.normal(0.0, 5.0, size=n_questions)
        noise_b = rng.normal(0.0, 5.0, size=n_questions)

        scores_a = np.clip(base_scores + true_difference + noise_a, 0.0, 100.0)
        scores_b = np.clip(base_scores + noise_b, 0.0, 100.0)
        # True difference on this population
        trial_true_diff = float(np.mean(scores_a - scores_b))

        obs_diff, ci_low, ci_high = paired_bootstrap_ci(
            scores_a, scores_b, num_bootstrap=1000, alpha=0.05, seed=trial_idx + 1000
        )

        assert ci_low <= obs_diff <= ci_high
        if ci_low <= trial_true_diff <= ci_high:
            contained_count += 1

    assert contained_count >= 90, f"Expected >= 90/100 coverage, got {contained_count}/100"


def test_paired_bootstrap_ci_validation():
    # Empty inputs return zeros
    assert paired_bootstrap_ci([], []) == (0.0, 0.0, 0.0)

    # Mismatched lengths raise ValueError
    with pytest.raises(ValueError, match="equal lengths"):
        paired_bootstrap_ci([1.0, 2.0], [1.0])

    # Identical arrays yield exact 0 difference with [0, 0] CI
    diff, low, high = paired_bootstrap_ci([10.0, 20.0, 30.0], [10.0, 20.0, 30.0])
    assert diff == 0.0
    assert low == 0.0
    assert high == 0.0


def test_system_metrics_to_dict():
    m = SystemMetrics(
        system_name="TestSystem",
        setting="per_question",
        split="test",
        num_questions=100,
        recall_at_2=35.0,
        recall_at_5=50.0,
        later_hop_recall_at_5=30.0,
        support_f1=45.0,
        answer_f1=55.0,
        answer_em=40.0,
        cost_per_question=0.002,
        latency_p50_ms=450.0,
        latency_p95_ms=1200.0,
        per_question_scores={"recall_at_5": [50.0] * 100},
    )
    d = m.to_dict()
    assert d["system_name"] == "TestSystem"
    assert d["recall_at_5"] == 50.0
    # Raw per-question scores omitted from summary dict
    assert "per_question_scores" not in d


def test_format_markdown_table():
    systems = [
        SystemMetrics(
            system_name="BM25",
            setting="per_question",
            split="test",
            num_questions=300,
            recall_at_2=35.5,
            recall_at_5=49.8,
            later_hop_recall_at_5=32.1,
            support_f1=40.2,
            answer_f1=0.0,
            answer_em=0.0,
            cost_per_question=0.0,
            latency_p50_ms=12.5,
            latency_p95_ms=25.0,
        ),
        SystemMetrics(
            system_name="GraphRAG MVP",
            setting="per_question",
            split="test",
            num_questions=300,
            recall_at_2=52.0,
            recall_at_5=68.5,
            later_hop_recall_at_5=58.0,
            support_f1=61.2,
            answer_f1=42.0,
            answer_em=31.0,
            cost_per_question=0.015,
            latency_p50_ms=350.0,
            latency_p95_ms=900.0,
        ),
    ]

    table = format_markdown_table(systems)
    assert "| System | Setting | Split | R@2 (%) | R@5 (%) | Later-hop R@5 (%) | Support F1 (%) | Answer F1 (%) | Answer EM (%) | Cost/q ($) | Latency p50 (ms) | Latency p95 (ms) |" in table
    assert "| BM25 | per_question | test | 35.50 | 49.80 | 32.10 | 40.20 | 0.00 | 0.00 | $0.0000 | 12.5 | 25.0 |" in table
    assert "| GraphRAG MVP | per_question | test | 52.00 | 68.50 | 58.00 | 61.20 | 42.00 | 31.00 | $0.0150 | 350.0 | 900.0 |" in table


def test_compare_systems():
    sys_a = SystemMetrics(
        system_name="System A",
        setting="per_question",
        split="test",
        per_question_scores={"recall_at_5": [80.0, 90.0, 100.0, 70.0, 85.0]},
    )
    sys_b = SystemMetrics(
        system_name="System B",
        setting="per_question",
        split="test",
        per_question_scores={"recall_at_5": [50.0, 60.0, 70.0, 40.0, 55.0]},
    )

    comp = compare_systems(sys_a, sys_b, metric="recall_at_5", num_bootstrap=500, seed=42)
    assert comp["system_a"] == "System A"
    assert comp["system_b"] == "System B"
    assert comp["observed_diff"] == 30.0
    assert comp["ci_lower"] <= comp["observed_diff"] <= comp["ci_upper"]
    assert comp["is_paired"] is True


def test_parse_baseline_output_to_systems():
    raw_data = {
        "slice": "test_slice",
        "tune": {
            "per_question_bm25": {
                "question_count": 200,
                "recall_at_2": 34.0,
                "recall_at_5": 50.0,
                "later_hop_recall_at_5": 32.0,
                "support_f1": 42.0,
            },
            "pooled_bm25": {
                "question_count": 200,
                "recall_at_2": 31.0,
                "recall_at_5": 41.0,
                "later_hop_recall_at_5": 25.0,
                "support_f1": 35.0,
            },
            "random_5": {
                "question_count": 200,
                "recall_at_k": 25.0,
                "later_hop_recall_at_5": 24.5,
            },
        },
    }

    systems = parse_baseline_output_to_systems(raw_data)
    assert len(systems) == 3
    names = [s.system_name for s in systems]
    assert "BM25" in names
    assert "Random-5 Floor" in names

    md = build_full_report_markdown(
        report_title="Baseline Report v0",
        systems=systems,
        notes=["Real data baseline run on ans_dev_mvp_v1 slice."],
    )
    assert "# Baseline Report v0" in md
    assert "BM25" in md
    assert "Random-5 Floor" in md
