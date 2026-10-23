"""CI dataset resolution uses the fixture only when the full file is absent."""

from __future__ import annotations

import pytest

from tests.live.musique_ci import FIXTURE_DEV, FIXTURE_SLICE, musique_inputs


def test_require_live_uses_fixture_when_full_dataset_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REQUIRE_LIVE", "1")
    monkeypatch.delenv("MUSIQUE_USE_FIXTURE", raising=False)
    monkeypatch.setattr("tests.live.musique_ci.FULL_DEV", FIXTURE_DEV.with_name("missing.jsonl"))
    dataset, slice_path = musique_inputs()
    assert dataset == FIXTURE_DEV
    assert slice_path == FIXTURE_SLICE


def test_missing_dataset_skips_without_require_live(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REQUIRE_LIVE", raising=False)
    monkeypatch.delenv("MUSIQUE_USE_FIXTURE", raising=False)
    monkeypatch.setattr("tests.live.musique_ci.FULL_DEV", FIXTURE_DEV.with_name("missing.jsonl"))
    with pytest.raises(pytest.skip.Exception):
        musique_inputs()


def test_use_fixture_flag_ignores_the_full_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MUSIQUE_USE_FIXTURE", "1")
    dataset, slice_path = musique_inputs()
    assert dataset == FIXTURE_DEV
    assert slice_path == FIXTURE_SLICE
