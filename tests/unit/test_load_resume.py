"""Deterministic resume tests for ``run_requests`` (fake pipeline and fake Neo4j).

The fake graph maps ``chunk_id`` to a persisted-state string. The fake pipeline
writes to it, so tests can simulate a stop between chunks, a crash after the
Neo4j write but before the manifest checkpoint, and pre-existing partial data.
A "new process" is simulated by building a new ``ManifestStore`` over the same
manifest file and a new fake pipeline.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from graphrag_pipeline.extraction.schemas import ExtractionRequest, Provenance
from graphrag_pipeline.ingestion.load_musique_dataset import RunOutcome, run_requests
from graphrag_pipeline.ingestion.progress_manifest import ManifestStore
from graphrag_pipeline.ingestion.reconcile import ABSENT, COMPLETE, PARTIAL, ChunkState

MODULE = "graphrag_pipeline.ingestion.load_musique_dataset"


class Crash(BaseException):
    """Simulates sudden process death (not caught by ``except Exception``)."""


def make_request(record_id: str, idx: int, sha: str = "a" * 64) -> ExtractionRequest:
    return ExtractionRequest(
        chunk_id=f"musique:{record_id}:paragraph:{idx}",
        record_id=record_id,
        document_id="musique:document:example",
        paragraph_idx=idx,
        text=f"Paragraph {idx}.",
        title="Example",
        provenance=Provenance(
            source_dataset="musique",
            source_split="fixture",
            source_file="fixture.jsonl",
            source_sha256=sha,
            source_line=1,
        ),
    )


class FakeWorld:
    """Fake Neo4j state plus a fake pipeline that mutates it."""

    def __init__(self) -> None:
        self.graph: dict[str, str] = {}
        self.calls: list[str] = []
        self.fail_on: set[str] = set()
        self.crash_after_write_on: set[str] = set()
        self.crash_before_write_on: set[str] = set()
        self.write_nothing_on: set[str] = set()
        self.stop_after_calls: int | None = None

    def classify(self, _connection, request: ExtractionRequest) -> ChunkState:
        state = self.graph.get(request.chunk_id, ABSENT)
        return ChunkState(state, "" if state in (ABSENT, COMPLETE) else "fixture reason", ["doc-1"], ["chunk-1"])

    def pipeline(self) -> MagicMock:
        async def run_async(text, document_metadata):
            chunk_id = document_metadata["chunk_id"]
            self.calls.append(chunk_id)
            if chunk_id in self.crash_before_write_on:
                raise Crash()
            if chunk_id in self.fail_on:
                raise RuntimeError("provider failure")
            if chunk_id not in self.write_nothing_on:
                self.graph[chunk_id] = COMPLETE
            if chunk_id in self.crash_after_write_on:
                raise Crash()
            return SimpleNamespace(run_id=f"run-{len(self.calls)}")

        pipeline = MagicMock()
        pipeline.run_async = run_async
        pipeline.runner.config.llm_config = {"default": SimpleNamespace(root=SimpleNamespace(last_response=None))}
        return pipeline

    def stop_requested(self) -> bool:
        return self.stop_after_calls is not None and len(self.calls) >= self.stop_after_calls


def run(world: FakeWorld, requests, manifest_path: Path, **kwargs):
    """Run the loader once as a fresh 'process' over the shared fake world."""
    outcome = RunOutcome()
    store = ManifestStore(manifest_path=manifest_path)
    with (
        patch(f"{MODULE}.Neo4jConnection"),
        patch(f"{MODULE}.build_simple_kg_pipeline", return_value=world.pipeline()),
        patch(f"{MODULE}.classify_chunk", side_effect=world.classify),
        patch(f"{MODULE}.synchronize_chunk_provenance"),
        patch(f"{MODULE}.save_invalid_llm_response", return_value=None),
    ):
        result = asyncio.run(
            run_requests(
                requests,
                MagicMock(),
                MagicMock(),
                show_progress=False,
                store=store,
                outcome=outcome,
                **kwargs,
            )
        )
    return result, outcome, store


def four_chunks() -> list[ExtractionRequest]:
    return [make_request("r1", 0), make_request("r1", 1), make_request("r2", 0), make_request("r2", 1)]


def test_clean_stop_then_new_process_skips_verified_and_resumes_pending(tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    requests = four_chunks()
    world = FakeWorld()
    world.stop_after_calls = 3  # stop before starting the 4th chunk

    (succeeded, failed), outcome, store = run(world, requests, path, stop_requested=world.stop_requested)

    assert len(succeeded) == 3 and not failed
    assert outcome.stopped_early and outcome.not_attempted == 1
    assert store.complete_record_ids(requests) == {"r1"}  # r2 only half done

    # Inspect the durable checkpoint before opening another store: reloading
    # demotes in-flight entries and could otherwise hide a clean-stop bug.
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert {chunk_id: chunk["status"] for chunk_id, chunk in persisted["chunks"].items()} == {
        "musique:r1:paragraph:0": "verified",
        "musique:r1:paragraph:1": "verified",
        "musique:r2:paragraph:0": "verified",
        "musique:r2:paragraph:1": "pending",
    }
    assert persisted["chunks"]["musique:r2:paragraph:1"]["attempts"] == 0

    # "New process": new store/pipeline, same source, no stop.
    world.calls.clear()
    world.stop_after_calls = None
    (succeeded, failed), outcome, store = run(world, requests, path)

    assert world.calls == ["musique:r2:paragraph:1"]  # only the pending chunk hit the provider
    assert len(outcome.skipped_verified) == 3
    assert not outcome.stopped_early
    assert store.complete_record_ids(requests) == {"r1", "r2"}
    persisted_after_resume = json.loads(path.read_text(encoding="utf-8"))
    assert all(chunk["status"] == "verified" for chunk in persisted_after_resume["chunks"].values())
    # Resuming pending work must not increment attempts for skipped chunks.
    assert all(chunk["attempts"] == 1 for chunk in persisted_after_resume["chunks"].values())


def test_failed_chunk_never_completes_record_and_only_it_is_retried_on_resume(tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    requests = four_chunks()
    world = FakeWorld()
    world.fail_on = {"musique:r1:paragraph:1"}

    (succeeded, failed), outcome, store = run(world, requests, path)

    assert [item["chunk_id"] for item in failed] == ["musique:r1:paragraph:1"]
    assert store.complete_record_ids(requests) == {"r2"}

    world.fail_on = set()
    world.calls.clear()
    (succeeded, failed), outcome, store = run(world, requests, path)

    assert world.calls == ["musique:r1:paragraph:1"]
    assert store.complete_record_ids(requests) == {"r1", "r2"}


def test_crash_after_neo4j_write_before_checkpoint_is_verified_not_rewritten(tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    requests = [make_request("r1", 0), make_request("r1", 1)]
    world = FakeWorld()
    world.crash_after_write_on = {"musique:r1:paragraph:1"}

    try:
        run(world, requests, path)
    except Crash:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected simulated crash")

    assert world.graph["musique:r1:paragraph:1"] == COMPLETE  # Neo4j has it...
    world.crash_after_write_on = set()
    world.calls.clear()

    (_, failed), outcome, store = run(world, requests, path)  # ...but the manifest did not.

    assert world.calls == []  # no provider call, no duplicate write
    assert outcome.interrupted_in_flight == ["musique:r1:paragraph:1"]
    assert "musique:r1:paragraph:1" in outcome.skipped_verified
    assert not failed
    assert store.complete_record_ids(requests) == {"r1"}


def test_crash_before_write_leaves_chunk_pending_until_neo4j_is_checked(tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    requests = [make_request("r1", 0)]
    world = FakeWorld()
    world.crash_before_write_on = {"musique:r1:paragraph:0"}

    try:
        run(world, requests, path)
    except Crash:
        pass

    world.crash_before_write_on = set()
    world.calls.clear()
    (_, failed), outcome, store = run(world, requests, path)

    assert world.calls == ["musique:r1:paragraph:0"]  # nothing in Neo4j, so it is re-ingested
    assert store.complete_record_ids(requests) == {"r1"}


def test_existing_partial_chunk_is_reported_and_skipped_without_provider_call(tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    requests = [make_request("r1", 0), make_request("r1", 1)]
    world = FakeWorld()
    world.graph["musique:r1:paragraph:0"] = PARTIAL

    (succeeded, failed), outcome, store = run(world, requests, path)

    assert world.calls == ["musique:r1:paragraph:1"]
    assert [item["chunk_id"] for item in outcome.needs_investigation] == ["musique:r1:paragraph:0"]
    report = outcome.needs_investigation[0]
    assert report["state"] == PARTIAL and report["document_element_ids"] == ["doc-1"]
    assert store.complete_record_ids(requests) == set()  # record never reported as loaded
    assert world.graph["musique:r1:paragraph:0"] == PARTIAL  # untouched


def test_chunk_not_marked_verified_when_post_write_check_fails(tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    requests = [make_request("r1", 0)]
    world = FakeWorld()
    world.write_nothing_on = {"musique:r1:paragraph:0"}  # pipeline "succeeds" but writes nothing

    (succeeded, failed), outcome, store = run(world, requests, path)

    assert not succeeded and len(failed) == 1
    assert "post-write Neo4j verification failed" in str(failed[0])
    assert store.complete_record_ids(requests) == set()


def test_verified_manifest_entry_is_reingested_if_neo4j_data_was_removed(tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    requests = [make_request("r1", 0)]
    world = FakeWorld()
    run(world, requests, path)
    world.graph.clear()  # operator deleted the nodes after investigating
    world.calls.clear()

    _, _, store = run(world, requests, path)

    assert world.calls == ["musique:r1:paragraph:0"]
    assert store.complete_record_ids(requests) == {"r1"}


def test_changed_dataset_checksum_does_not_reuse_old_progress(tmp_path: Path) -> None:
    import pytest

    from graphrag_pipeline.ingestion.progress_manifest import ManifestSourceMismatch

    path = tmp_path / "m.json"
    world = FakeWorld()
    run(world, [make_request("r1", 0, sha="a" * 64)], path)
    world.calls.clear()

    with pytest.raises(ManifestSourceMismatch):
        run(world, [make_request("r1", 0, sha="b" * 64)], path)
    assert world.calls == []
