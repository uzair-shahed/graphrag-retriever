"""Unit tests for the durable ingestion progress manifest (no live services)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from graphrag_pipeline.extraction.schemas import ExtractionRequest, Provenance
from graphrag_pipeline.ingestion.progress_manifest import (
    PENDING,
    VERIFIED,
    ManifestError,
    ManifestSourceMismatch,
    ManifestStore,
    ProgressManifest,
    default_manifest_path,
)


def make_request(
    record_id: str, paragraph_idx: int, sha: str = "a" * 64, split: str = "full_dev"
) -> ExtractionRequest:
    return ExtractionRequest(
        chunk_id=f"musique:{record_id}:paragraph:{paragraph_idx}",
        record_id=record_id,
        document_id="musique:document:example",
        paragraph_idx=paragraph_idx,
        text=f"Paragraph {paragraph_idx} of {record_id}.",
        title="Example",
        provenance=Provenance(
            source_dataset="musique",
            source_split=split,
            source_file="fixture.jsonl",
            source_sha256=sha,
            source_line=1,
        ),
    )


def open_manifest(path: Path, sha: str = "a" * 64, split: str = "full_dev", **kwargs) -> ProgressManifest:
    return ProgressManifest.open(path, sha, split, **kwargs)


def test_record_is_complete_only_when_every_expected_chunk_is_verified(tmp_path: Path) -> None:
    manifest = open_manifest(tmp_path / "m.json")
    manifest.register([make_request("r1", 0), make_request("r1", 1)])

    assert not manifest.is_record_complete("r1")
    manifest.mark_verified("musique:r1:paragraph:0")
    assert not manifest.is_record_complete("r1")
    manifest.mark_verified("musique:r1:paragraph:1")
    assert manifest.is_record_complete("r1")


def test_failed_or_investigated_chunk_keeps_record_incomplete(tmp_path: Path) -> None:
    manifest = open_manifest(tmp_path / "m.json")
    manifest.register([make_request("r1", 0), make_request("r1", 1)])
    manifest.mark_verified("musique:r1:paragraph:0")
    manifest.mark_failed("musique:r1:paragraph:1", {"error": "boom"})
    assert not manifest.is_record_complete("r1")

    manifest.mark_needs_investigation("musique:r1:paragraph:1", {"reason": "partial"})
    assert not manifest.is_record_complete("r1")


def test_state_persists_across_reload(tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    manifest = open_manifest(path)
    manifest.register([make_request("r1", 0)])
    manifest.mark_verified("musique:r1:paragraph:0")

    reloaded = open_manifest(path)
    assert reloaded.status("musique:r1:paragraph:0") == VERIFIED
    assert reloaded.is_record_complete("r1")


def test_in_flight_chunk_becomes_pending_and_is_reported_on_reload(tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    manifest = open_manifest(path)
    manifest.register([make_request("r1", 0)])
    manifest.mark_in_flight("musique:r1:paragraph:0")
    assert json.loads(path.read_text())["chunks"]["musique:r1:paragraph:0"]["status"] == "in_flight"

    reloaded = open_manifest(path)
    assert reloaded.status("musique:r1:paragraph:0") == PENDING
    assert reloaded.interrupted_in_flight == ["musique:r1:paragraph:0"]
    assert not reloaded.is_record_complete("r1")


def test_atomic_write_failure_leaves_previous_manifest_intact(tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    manifest = open_manifest(path)
    manifest.register([make_request("r1", 0)])
    before = path.read_text()

    with patch("graphrag_pipeline.ingestion.progress_manifest.os.replace", side_effect=OSError("disk")):
        with pytest.raises(ManifestError):
            manifest.mark_verified("musique:r1:paragraph:0")

    assert path.read_text() == before
    assert [p.name for p in tmp_path.iterdir()] == ["m.json"]  # no stray temp file
    assert open_manifest(path).status("musique:r1:paragraph:0") == PENDING


def test_checksum_mismatch_is_rejected_not_applied(tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    manifest = open_manifest(path, sha="a" * 64)
    manifest.register([make_request("r1", 0)])
    manifest.mark_verified("musique:r1:paragraph:0")

    with pytest.raises(ManifestSourceMismatch):
        open_manifest(path, sha="b" * 64)
    with pytest.raises(ManifestSourceMismatch):
        open_manifest(path, split="other_split")


def test_default_path_separates_sources_by_checksum_and_split(tmp_path: Path) -> None:
    one = default_manifest_path("a" * 64, "full_dev", tmp_path)
    assert one != default_manifest_path("b" * 64, "full_dev", tmp_path)
    assert one != default_manifest_path("a" * 64, "full_train", tmp_path)


def test_fresh_manifest_moves_old_file_aside_without_deleting(tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    manifest = open_manifest(path)
    manifest.register([make_request("r1", 0)])
    manifest.mark_verified("musique:r1:paragraph:0")

    fresh = open_manifest(path, fresh=True)
    assert not fresh.is_record_complete("r1")
    superseded = [p for p in tmp_path.iterdir() if "superseded" in p.name]
    assert len(superseded) == 1
    assert json.loads(superseded[0].read_text())["chunks"]


def test_corrupt_manifest_is_an_explicit_error(tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ManifestError):
        open_manifest(path)


def test_register_rejects_different_expected_chunks_for_same_record(tmp_path: Path) -> None:
    manifest = open_manifest(tmp_path / "m.json")
    manifest.register([make_request("r1", 0), make_request("r1", 1)])
    with pytest.raises(ManifestError):
        manifest.register([make_request("r1", 0)])


def test_store_uses_one_manifest_per_source_and_explicit_path_is_single_source(tmp_path: Path) -> None:
    store = ManifestStore(state_dir=tmp_path)
    a = store.for_request(make_request("r1", 0, sha="a" * 64))
    b = store.for_request(make_request("r2", 0, sha="b" * 64))
    assert a.path != b.path

    explicit = ManifestStore(manifest_path=tmp_path / "one.json")
    explicit.for_request(make_request("r1", 0, sha="a" * 64))
    with pytest.raises(ManifestError):
        explicit.for_request(make_request("r2", 0, sha="b" * 64))


def test_explicit_path_for_a_different_source_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "one.json"
    first = ManifestStore(manifest_path=path)
    first.register([make_request("r1", 0, sha="a" * 64)])

    second = ManifestStore(manifest_path=path)
    with pytest.raises(ManifestSourceMismatch):
        second.register([make_request("r1", 0, sha="b" * 64)])
    assert os.path.exists(path)

