"""Choose the MuSiQue file live tests read.

CI does not have the full dataset. When ``REQUIRE_LIVE=1`` and the full file is
absent, tests use the committed fixture. ``MUSIQUE_USE_FIXTURE=1`` forces the
fixture even if the full file is on disk, which is how the CI path is checked
locally without touching the real database's expected file.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

FULL_DEV = Path("dataset/musique/data/data/musique_ans_v1.0_dev.jsonl")
FULL_SLICE = Path("eval/slices/ans_dev_mvp_v1.json")
FIXTURE_DEV = Path("tests/fixtures/musique_ci.jsonl")
FIXTURE_SLICE = Path("tests/fixtures/musique_ci_slice.json")


def musique_inputs() -> tuple[Path, Path]:
    """Return ``(jsonl, slice_json)`` or skip/fail according to REQUIRE_LIVE."""
    use_fixture = os.environ.get("MUSIQUE_USE_FIXTURE") == "1"
    if use_fixture:
        if not (FIXTURE_DEV.exists() and FIXTURE_SLICE.exists()):
            pytest.fail(f"CI fixture missing at {FIXTURE_DEV}")
        return FIXTURE_DEV, FIXTURE_SLICE
    if FULL_DEV.exists():
        return FULL_DEV, FULL_SLICE
    if os.environ.get("REQUIRE_LIVE") == "1" and FIXTURE_DEV.exists() and FIXTURE_SLICE.exists():
        return FIXTURE_DEV, FIXTURE_SLICE
    if os.environ.get("REQUIRE_LIVE") == "1":
        pytest.fail(f"MuSiQue dataset missing at {FULL_DEV} and CI fixture missing at {FIXTURE_DEV}")
    pytest.skip(f"MuSiQue dataset not found at {FULL_DEV}")
