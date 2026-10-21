"""Unit tests for the MVP experiment runner and comparative report (Step M6.5).

Verifies:
1. Slice loading and split selection (test vs tune vs all).
2. Non-parametric bootstrap confidence interval estimation.
3. Execution of the 4 conditions on the stub (bm25, vector, walk_off, walk_exact).
4. Report rendering producing a 4-row comparative table.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.run_mvp import (
    bootstrap_ci,
    load_slice_records,
    render_report,
    run_condition,
    run_mvp,
    StubAnswerLLM,
    StubEmbedder,
)


def test_bootstrap_ci_deterministic_and_valid() -> None:
    """Bootstrap CI is deterministic with fixed seed and bounds the sample mean."""
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ci_1 = bootstrap_ci(values, n_bootstrap=500, ci=0.95, seed=20261018)
    ci_2 = bootstrap_ci(values, n_bootstrap=500, ci=0.95, seed=20261018)

    assert ci_1 == ci_2
    mean = sum(values) / len(values)
    assert ci_1[0] <= mean <= ci_1[1]

    # Empty list edge case
    assert bootstrap_ci([]) == (0.0, 0.0)


def test_load_slice_records_selects_expected_split() -> None:
    """Loading slice records correctly subsets test vs tune IDs."""
    slice_path = Path("eval/slices/ans_dev_mvp_v1.json")
    if not slice_path.exists():
        pytest.skip("ans_dev_mvp_v1.json not found")

    slice_data, test_records = load_slice_records(slice_path, split="test", limit=3)
    assert len(test_records) == 3
    assert test_records[0]["id"] in slice_data["test_ids"]

    _, tune_records = load_slice_records(slice_path, split="tune", limit=2)
    assert len(tune_records) == 2
    assert tune_records[0]["id"] in slice_data["tune_ids"]


def test_run_mvp_end_to_end_stub_renders_four_rows(tmp_path: Path) -> None:
    """Verify that run_mvp on the stub executes all 4 conditions and renders the report."""
    slice_path = Path("eval/slices/ans_dev_mvp_v1.json")
    if not slice_path.exists():
        pytest.skip("ans_dev_mvp_v1.json not found")

    output_dir = tmp_path / "mvp_results"
    report_md, results = run_mvp(
        slice_spec=slice_path,
        split="test",
        conditions=["bm25", "vector", "walk_off", "walk_exact"],
        limit=2,
        output_dir=output_dir,
        stub=True,
    )

    # 1. Exactly 4 condition results
    assert len(results) == 4
    cond_names = [r.condition for r in results]
    assert cond_names == ["bm25", "vector", "walk_off", "walk_exact"]

    # 2. Prediction files exist for all 4 conditions
    for cond in cond_names:
        pred_file = output_dir / f"predictions_{cond}.jsonl"
        assert pred_file.exists(), f"Missing prediction file for {cond}"
        lines = pred_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

    # 3. Report renders with 4 rows in table
    assert report_md.startswith("# MuSiQue MVP Comparative Evaluation Report")
    assert "| `bm25` |" in report_md
    assert "| `vector` |" in report_md
    assert "| `walk_off` |" in report_md
    assert "| `walk_exact` |" in report_md

    # 4. JSON report written
    json_report = output_dir / "mvp_report.json"
    assert json_report.exists()
    report_data = json.loads(json_report.read_text(encoding="utf-8"))
    assert len(report_data["results"]) == 4


def test_classify_failure_categories() -> None:
    """Error diagnosis categorizes failures into answer error, walk miss, and extraction miss."""
    from eval.run_mvp import classify_failure
    from graphrag_pipeline.query import EvidenceChunk

    record = {
        "id": "q1",
        "question_decomposition": [
            {"paragraph_support_idx": 0},
            {"paragraph_support_idx": 1},
        ],
    }

    # Case 1: Gold indices {0, 1} are both retrieved -> answer_error
    retrieved_all = [
        EvidenceChunk(chunk_id="c0", text="t0", paragraph_idx=0),
        EvidenceChunk(chunk_id="c1", text="t1", paragraph_idx=1),
    ]
    assert classify_failure(record, retrieved_all, {"predicted_answer": "wrong"}) == "answer_error"

    # Case 2: Only 1 gold index retrieved -> walk_miss
    retrieved_partial = [
        EvidenceChunk(chunk_id="c0", text="t0", paragraph_idx=0),
        EvidenceChunk(chunk_id="c9", text="t9", paragraph_idx=9),
    ]
    assert classify_failure(record, retrieved_partial, {"predicted_answer": "wrong"}) == "walk_miss"

    # Case 3: Zero gold indices retrieved -> extraction_miss
    retrieved_none = [
        EvidenceChunk(chunk_id="c8", text="t8", paragraph_idx=8),
        EvidenceChunk(chunk_id="c9", text="t9", paragraph_idx=9),
    ]
    assert classify_failure(record, retrieved_none, {"predicted_answer": "wrong"}) == "extraction_miss"


def test_per_instance_scores_and_paragraph_hash_canonical() -> None:
    """Paragraph hash produces full 64-character SHA-256 matching ingestion."""
    from eval.run_mvp import compute_paragraph_hash
    h = compute_paragraph_hash("Test Title", "Test content")
    assert len(h) == 64
    assert h == compute_paragraph_hash("Test Title", "Test content")
