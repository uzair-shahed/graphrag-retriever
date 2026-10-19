"""Prediction formatting, validation, and official MuSiQue scorer invocation."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REQUIRED_PREDICTION_FIELDS = {
    "id",
    "predicted_answer",
    "predicted_support_idxs",
    "predicted_answerable",
}


def validate_prediction(prediction: dict[str, Any]) -> None:
    """Validate that one prediction record conforms to the official MuSiQue contract."""
    missing = REQUIRED_PREDICTION_FIELDS - set(prediction.keys())
    if missing:
        raise ValueError(f"Prediction missing required fields: {sorted(missing)}")
    if not isinstance(prediction["id"], str) or not prediction["id"].strip():
        raise ValueError("Prediction 'id' must be a non-empty string")
    if not isinstance(prediction["predicted_answer"], str):
        raise ValueError("Prediction 'predicted_answer' must be a string")
    if not isinstance(prediction["predicted_support_idxs"], list):
        raise ValueError("Prediction 'predicted_support_idxs' must be a list of integer indices")
    for idx in prediction["predicted_support_idxs"]:
        if not isinstance(idx, int):
            raise ValueError(f"Invalid non-integer support index: {idx!r}")
    if not isinstance(prediction["predicted_answerable"], bool):
        raise ValueError("Prediction 'predicted_answerable' must be a boolean")


def write_predictions(
    predictions: list[dict[str, Any]],
    output_path: Path,
    gold_records: list[dict[str, Any]] | None = None,
) -> Path:
    """Validate and write predictions in exact gold-record count and order."""
    for p in predictions:
        validate_prediction(p)

    if gold_records is not None:
        if len(predictions) != len(gold_records):
            raise ValueError(
                f"Prediction count mismatch: expected {len(gold_records)}, got {len(predictions)}"
            )
        for pred, gold in zip(predictions, gold_records):
            if pred["id"] != gold["id"]:
                raise ValueError(
                    f"Prediction ID order mismatch: expected {gold['id']}, got {pred['id']}"
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    return output_path


def load_predictions(path: Path) -> list[dict[str, Any]]:
    """Load and validate predictions from JSONL file."""
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found at {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            validate_prediction(item)
            records.append(item)
    return records


def evaluate_predictions(
    predictions_path: Path,
    gold_path: Path,
    third_party_dir: Path | None = None,
) -> dict[str, float]:
    """Execute the official MuSiQue scorer on predictions and gold JSONL."""
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold records file not found: {gold_path}")

    scorer_dir = (third_party_dir or Path("third_party/musique")).resolve()
    scorer_script = scorer_dir / "evaluate_v1.0.py"
    if not scorer_script.exists():
        raise FileNotFoundError(f"Official scorer script not found: {scorer_script}")

    # Ensure third_party/musique is in sys.path so 'metrics' imports succeed
    scorer_dir_str = str(scorer_dir)
    if scorer_dir_str not in sys.path:
        sys.path.insert(0, scorer_dir_str)

    spec = importlib.util.spec_from_file_location("musique_official_evaluate", scorer_script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec for {scorer_script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["musique_official_evaluate"] = module
    spec.loader.exec_module(module)

    raw_metrics: dict[str, Any] = module.evaluate(str(predictions_path), str(gold_path))
    return {k: float(v) for k, v in raw_metrics.items()}
