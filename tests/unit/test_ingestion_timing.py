"""Tests for the ingestion timing helpers (no live services)."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import patch

from graphrag_pipeline.ingestion.load_musique_dataset import (
    format_duration,
    load_dataset,
    measure_duration_seconds,
)


def test_format_duration_seconds_only() -> None:
    """Durations under a minute render as plain seconds."""
    assert format_duration(12.345) == "12.35s"


def test_format_duration_minutes_and_seconds() -> None:
    """Durations of a minute or more render as minutes plus seconds."""
    assert format_duration(125.4) == "2m 5.4s"


def test_format_duration_hours_minutes_and_seconds() -> None:
    """Durations of an hour or more render as hours, minutes, and seconds."""
    assert format_duration(3725.0) == "1h 2m 5.0s"


def test_measure_duration_seconds_uses_perf_counter_delta() -> None:
    """The measured duration is the difference between two perf_counter reads."""
    with patch("time.perf_counter", side_effect=[10.0, 14.5]):
        start = time.perf_counter()
        elapsed = measure_duration_seconds(start)

    assert elapsed == 4.5


def test_load_dataset_dry_run_reports_collect_and_elapsed_time(tmp_path: Path) -> None:
    """A dry run measures collection time and overall elapsed time, but no run time."""
    dataset_path = tmp_path / "musique_full_v1.0_dev.jsonl"
    dataset_path.write_text(
        '{"id": "record-1", "paragraphs": [{"idx": 0, "title": "T", "paragraph_text": "Text."}]}\n',
        encoding="utf-8",
    )

    summary = asyncio.run(
        load_dataset(
            dataset_path=dataset_path,
            start_record=1,
            max_records=1,
            source_split=None,
            dry_run=True,
        )
    )

    assert summary.collect_seconds >= 0
    assert summary.elapsed_seconds >= summary.collect_seconds
    assert summary.run_seconds == 0.0
    assert "elapsed" in summary.to_dict()
    assert summary.elapsed == summary.to_dict()["elapsed"]
