"""Unit tests for the seeded, hop-stratified MuSiQue evaluation slice."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import pytest

from scripts.make_slice import create_slice


def test_make_slice_deterministic_counts_and_hops(tmp_path: Path) -> None:
    """Evaluation slice must be 100% deterministic and match measured counts."""
    input_file = Path("dataset/musique/data/data/musique_ans_v1.0_dev.jsonl")
    if not input_file.exists():
        pytest.skip("MuSiQue Ans dev dataset not downloaded locally.")

    output_1 = tmp_path / "slice_1.json"
    output_2 = tmp_path / "slice_2.json"

    data_1 = create_slice(input_file, output_1, seed=20261018, size=500, tune_size=200)
    data_2 = create_slice(input_file, output_2, seed=20261018, size=500, tune_size=200)

    # Identical IDs across repeated runs
    assert data_1["tune_ids"] == data_2["tune_ids"]
    assert data_1["test_ids"] == data_2["test_ids"]

    # Stratified hop counts: 259 / 157 / 84
    assert data_1["hop_distribution"]["total"] == {"2": 259, "3": 157, "4": 84}
    assert data_1["hop_distribution"]["tune"] == {"2": 104, "3": 63, "4": 33}
    assert data_1["hop_distribution"]["test"] == {"2": 155, "3": 94, "4": 51}

    # Exactly 6,838 unique paragraphs
    assert data_1["unique_paragraphs"] == 6838
    assert len(data_1["tune_ids"]) == 200
    assert len(data_1["test_ids"]) == 300


def test_committed_evaluation_slice_matches_spec() -> None:
    """The generated eval/slices/ans_dev_mvp_v1.json matches the specification."""
    slice_path = Path("eval/slices/ans_dev_mvp_v1.json")
    assert slice_path.exists(), "eval/slices/ans_dev_mvp_v1.json must exist"

    with slice_path.open(encoding="utf-8") as f:
        data = json.load(f)

    assert data["size"] == 500
    assert data["tune_size"] == 200
    assert data["test_size"] == 300
    assert data["hop_distribution"]["total"] == {"2": 259, "3": 157, "4": 84}
    assert data["unique_paragraphs"] == 6838


def test_inference_modules_never_read_hop_count_from_id() -> None:
    """Inference code must never read hop count from question IDs."""
    inference_files = [
        Path("src/graphrag_pipeline/retrieval.py"),
        Path("src/graphrag_pipeline/answer.py"),
        Path("src/graphrag_pipeline/query.py"),
        Path("src/graphrag_pipeline/query_command.py"),
        Path("src/graphrag_pipeline/extraction/pipeline.py"),
        Path("src/graphrag_pipeline/ingestion/record_adapter.py"),
    ]
    for file_path in inference_files:
        code = file_path.read_text(encoding="utf-8")
        assert "split('hop')" not in code and 'split("hop")' not in code, (
            f"{file_path} must not inspect hop count from record ID"
        )
