"""Paired MVP differences align questions and bootstrap the mean gap."""

from __future__ import annotations

import json
from pathlib import Path

from eval.paired_mvp import compare_runs, question_scores, render_paired_markdown


def _gold(question_id: str, answer: str) -> dict:
    return {
        "id": question_id,
        "answer": answer,
        "answer_aliases": [],
        "paragraphs": [
            {"idx": 0, "is_supporting": True},
            {"idx": 1, "is_supporting": False},
        ],
    }


def _pred(question_id: str, answer: str, support: list[int]) -> dict:
    return {"id": question_id, "predicted_answer": answer, "predicted_support_idxs": support}


def test_question_scores_exact_match_and_empty_prediction() -> None:
    gold = _gold("q1", "Paris")
    perfect = question_scores(_pred("q1", "Paris", [0]), gold)
    empty = question_scores(_pred("q1", "", []), gold)
    assert perfect["answer_em"] == 1.0
    assert perfect["answer_f1"] == 1.0
    assert perfect["support_f1"] == 1.0
    assert empty["answer_em"] == 0.0
    assert empty["answer_f1"] == 0.0
    assert empty["support_f1"] == 0.0


def test_paired_difference_is_question_aligned() -> None:
    gold = [_gold("q1", "Paris"), _gold("q2", "Rome")]
    left = [_pred("q2", "Rome", [0]), _pred("q1", "Paris", [0])]
    right = [_pred("q1", "", []), _pred("q2", "Rome", [0])]
    report = compare_runs(gold, {"left": left, "right": right}, pairs=[("left", "right")])
    answer_f1 = report["comparisons"][0]["metrics"][0]
    assert answer_f1["metric"] == "answer_f1"
    assert answer_f1["n"] == 2
    assert answer_f1["higher"] == 1
    assert answer_f1["tie"] == 1
    assert answer_f1["mean"] == 0.5
    assert answer_f1["ci_low"] <= answer_f1["mean"] <= answer_f1["ci_high"]


def test_identical_predictions_have_zero_difference(tmp_path: Path) -> None:
    gold = [_gold("q1", "Paris")]
    rows = [_pred("q1", "Paris", [0])]
    report = compare_runs(gold, {"a": rows, "b": rows}, pairs=[("a", "b")])
    for metric in report["comparisons"][0]["metrics"]:
        assert metric["mean"] == 0.0
        assert metric["tie"] == 1
        assert metric["separates"] is False
    rendered = render_paired_markdown(report)
    assert "a` minus `b" in rendered
    (tmp_path / "sample.json").write_text(json.dumps(report), encoding="utf-8")
