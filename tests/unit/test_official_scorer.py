"""Unit tests for the official MuSiQue evaluation scorer and prediction writer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest

from eval.predictions import (
    evaluate_predictions,
    load_predictions,
    validate_prediction,
    write_predictions,
)

EXPECTED_SCORER_SHA256 = "f5fe66ae61dbea5172cba9d428d9924a5811f3457edc945fc1d81369c30e74b7"


def sample_gold_records() -> list[dict]:
    return [
        {
            "id": "2hop__test_001",
            "question": "Who founded Company A?",
            "answer": "Alice Smith",
            "answer_aliases": ["Alice", "A. Smith"],
            "answerable": True,
            "paragraphs": [
                {"idx": 0, "title": "Doc 1", "paragraph_text": "Alice Smith founded Company A.", "is_supporting": True},
                {"idx": 1, "title": "Doc 2", "paragraph_text": "Company A was founded in 2010.", "is_supporting": True},
                {"idx": 2, "title": "Doc 3", "paragraph_text": "Bob works elsewhere.", "is_supporting": False},
            ],
        },
        {
            "id": "2hop__test_002",
            "question": "Where was Bob born?",
            "answer": "London",
            "answer_aliases": ["Greater London"],
            "answerable": True,
            "paragraphs": [
                {"idx": 0, "title": "Doc 1", "paragraph_text": "Bob was born in London.", "is_supporting": True},
                {"idx": 1, "title": "Doc 2", "paragraph_text": "London is a city.", "is_supporting": False},
            ],
        },
    ]


def test_scorer_script_checksum_is_unmodified() -> None:
    """The vendored official scorer must remain strictly unmodified."""
    scorer_path = Path("third_party/musique/evaluate_v1.0.py")
    assert scorer_path.exists(), "third_party/musique/evaluate_v1.0.py must exist"

    h = hashlib.sha256()
    h.update(scorer_path.read_bytes())
    actual_sha = h.hexdigest()
    assert actual_sha == EXPECTED_SCORER_SHA256, (
        f"Official scorer checksum mismatch: {actual_sha} != {EXPECTED_SCORER_SHA256}"
    )


def test_perfect_gold_predictions_score_100(tmp_path: Path) -> None:
    """Gold predictions must score 100 across answer and support metrics."""
    gold_records = sample_gold_records()
    gold_file = tmp_path / "gold.jsonl"
    with gold_file.open("w", encoding="utf-8") as f:
        for r in gold_records:
            f.write(json.dumps(r) + "\n")

    predictions = [
        {
            "id": "2hop__test_001",
            "predicted_answer": "Alice Smith",
            "predicted_support_idxs": [0, 1],
            "predicted_answerable": True,
        },
        {
            "id": "2hop__test_002",
            "predicted_answer": "London",
            "predicted_support_idxs": [0],
            "predicted_answerable": True,
        },
    ]
    pred_file = tmp_path / "predictions.jsonl"
    write_predictions(predictions, pred_file, gold_records=gold_records)

    metrics = evaluate_predictions(pred_file, gold_file)
    assert metrics["answer_f1"] == 1.0
    assert metrics["answer_em"] == 1.0
    assert metrics["support_f1"] == 1.0


def test_empty_predictions_score_zero(tmp_path: Path) -> None:
    """Empty predictions must score 0.0 across metrics."""
    gold_records = sample_gold_records()
    gold_file = tmp_path / "gold.jsonl"
    with gold_file.open("w", encoding="utf-8") as f:
        for r in gold_records:
            f.write(json.dumps(r) + "\n")

    predictions = [
        {
            "id": "2hop__test_001",
            "predicted_answer": "completely wrong answer",
            "predicted_support_idxs": [2],
            "predicted_answerable": True,
        },
        {
            "id": "2hop__test_002",
            "predicted_answer": "Paris",
            "predicted_support_idxs": [1],
            "predicted_answerable": True,
        },
    ]
    pred_file = tmp_path / "predictions.jsonl"
    write_predictions(predictions, pred_file, gold_records=gold_records)

    metrics = evaluate_predictions(pred_file, gold_file)
    assert metrics["answer_f1"] == 0.0
    assert metrics["answer_em"] == 0.0
    assert metrics["support_f1"] == 0.0


def test_shuffled_prediction_order_raises_error(tmp_path: Path) -> None:
    """Prediction order mismatch against gold records must raise an explicit error."""
    gold_records = sample_gold_records()
    gold_file = tmp_path / "gold.jsonl"
    with gold_file.open("w", encoding="utf-8") as f:
        for r in gold_records:
            f.write(json.dumps(r) + "\n")

    shuffled_predictions = [
        {
            "id": "2hop__test_002",
            "predicted_answer": "London",
            "predicted_support_idxs": [0],
            "predicted_answerable": True,
        },
        {
            "id": "2hop__test_001",
            "predicted_answer": "Alice Smith",
            "predicted_support_idxs": [0, 1],
            "predicted_answerable": True,
        },
    ]

    # write_predictions checks against gold_records if passed
    with pytest.raises(ValueError, match="Prediction ID order mismatch"):
        write_predictions(shuffled_predictions, tmp_path / "bad.jsonl", gold_records=gold_records)

    # Scorer itself asserts same order
    pred_file = tmp_path / "shuffled.jsonl"
    write_predictions(shuffled_predictions, pred_file)
    with pytest.raises(AssertionError, match="The instances.*should be in same order"):
        evaluate_predictions(pred_file, gold_file)


def test_validate_prediction_rejects_malformed_shapes() -> None:
    """Prediction validator must reject missing fields and wrong types."""
    with pytest.raises(ValueError, match="missing required fields"):
        validate_prediction({"id": "1"})

    with pytest.raises(ValueError, match="non-integer support index"):
        validate_prediction({
            "id": "1",
            "predicted_answer": "ans",
            "predicted_support_idxs": ["0"],  # string instead of int
            "predicted_answerable": True,
        })
