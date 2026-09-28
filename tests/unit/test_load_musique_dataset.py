"""Tests for the batch MuSiQue dataset loader CLI (dry-run path only).

These tests never call Neo4j or OpenRouter; they exercise the record-reading,
adaptation, and chunk-planning logic that runs before any live pipeline call.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from graphrag_pipeline.ingestion.load_musique_dataset import (
    derive_source_split,
    load_dataset,
)


def write_dataset(path: Path, records: list[dict]) -> None:
    """Write one JSON record per line to a temporary MuSiQue-shaped file."""
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def make_record(record_id: str, paragraph_count: int) -> dict:
    """Build a minimal raw MuSiQue record with the given number of paragraphs."""
    return {
        "id": record_id,
        "paragraphs": [
            {
                "idx": idx,
                "title": f"Title {record_id} {idx}",
                "paragraph_text": f"Text {idx}.",
                "is_supporting": False,
            }
            for idx in range(paragraph_count)
        ],
    }


def test_derive_source_split_from_official_file_name() -> None:
    """An official MuSiQue file name yields a variant_split provenance label."""
    assert derive_source_split(Path("dataset/musique/data/data/musique_full_v1.0_dev.jsonl")) == "full_dev"


def test_load_dataset_dry_run_plans_chunks_without_network(tmp_path: Path) -> None:
    """Dry-run planning counts every paragraph across the selected records."""
    dataset_path = tmp_path / "musique_full_v1.0_dev.jsonl"
    write_dataset(
        dataset_path,
        [make_record("record-1", paragraph_count=3), make_record("record-2", paragraph_count=2)],
    )

    summary = asyncio.run(
        load_dataset(
            dataset_path=dataset_path,
            start_record=1,
            max_records=2,
            source_split=None,
            dry_run=True,
        )
    )

    assert summary.records_read == 2
    assert summary.chunks_planned == 5
    assert summary.dead_lettered_records == []
    assert summary.source_split == "full_dev"


def test_load_dataset_dry_run_respects_start_record(tmp_path: Path) -> None:
    """Only the record at start_record onward is read and planned."""
    dataset_path = tmp_path / "musique_full_v1.0_dev.jsonl"
    write_dataset(
        dataset_path,
        [make_record("record-1", paragraph_count=2), make_record("record-2", paragraph_count=4)],
    )

    summary = asyncio.run(
        load_dataset(
            dataset_path=dataset_path,
            start_record=2,
            max_records=1,
            source_split=None,
            dry_run=True,
        )
    )

    assert summary.records_read == 1
    assert summary.chunks_planned == 4


def test_load_dataset_dry_run_dead_letters_malformed_record(tmp_path: Path) -> None:
    """A malformed line is dead-lettered while valid records are still planned."""
    dataset_path = tmp_path / "musique_full_v1.0_dev.jsonl"
    dataset_path.write_text(
        json.dumps(make_record("record-1", paragraph_count=1)) + "\n" + "{not valid json\n",
        encoding="utf-8",
    )

    summary = asyncio.run(
        load_dataset(
            dataset_path=dataset_path,
            start_record=1,
            max_records=2,
            source_split=None,
            dry_run=True,
        )
    )

    assert summary.records_read == 2
    assert summary.chunks_planned == 1
    assert len(summary.dead_lettered_records) == 1
