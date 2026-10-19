"""Scheduling and exact paragraph-based progress, independent of live services."""

import asyncio
from unittest.mock import MagicMock, patch

from graphrag_pipeline.ingestion import load_musique_dataset as loader
from graphrag_pipeline.ingestion.progress_manifest import ManifestStore
from graphrag_pipeline.ingestion.retry_queue import RetryStore
from graphrag_pipeline.ingestion.reconcile import ChunkState
from test_load_resume import FakeWorld, make_request, run


def execute(world, requests, tmp_path, retries, **kwargs):
    outcome = loader.RunOutcome()
    store = ManifestStore(state_dir=tmp_path / "m")
    with patch.object(loader, "Neo4jConnection"), patch.object(loader, "build_simple_kg_pipeline", return_value=world.pipeline()), patch.object(loader, "classify_chunk", side_effect=world.classify):
        succeeded, failed = asyncio.run(loader.run_requests(requests, MagicMock(), MagicMock(), store=store, retry_store=retries, outcome=outcome, show_progress=False, **kwargs))
    return succeeded, failed, outcome


def test_145_processed_five_failed_can_mean_only_four_complete_records(tmp_path):
    requests = [make_request(f"r{r}", i) for r in range(10) for i in range(20)]
    world = FakeWorld()
    world.fail_on = {make_request(f"r{r}", i).chunk_id for r, i in [(4, 0), (4, 1), (5, 0), (5, 1), (6, 0)]}
    world.stop_after_calls = 145
    (succeeded, failed), outcome, _ = run(world, requests, tmp_path / "m.json", stop_requested=world.stop_requested)
    counts = outcome.progress.counts()
    assert len(succeeded) == 140 and len(failed) == 5
    assert counts["chunks_processed"] == 145
    assert counts["records_complete"] == 4
    assert counts["records_partial"] == 4
    assert counts["records_unstarted"] == 2


def test_variable_paragraph_counts_and_interleaved_records(tmp_path):
    requests = [make_request("r1", 0), make_request("r2", 0), make_request("r1", 1), make_request("r3", 0), make_request("r3", 1), make_request("r1", 2)]
    (_, failed), outcome, _ = run(FakeWorld(), requests, tmp_path / "m.json")
    assert not failed
    assert outcome.progress.counts()["records_complete"] == 3
    assert outcome.progress.counts()["chunks_processed"] == 6


def test_reuse_updates_postfix_immediately_on_last_chunk(tmp_path):
    req = make_request("r1", 0)
    world = FakeWorld()
    world.graph[req.chunk_id] = "complete"
    with patch.object(loader, "tqdm") as bar:
        (_, failed), outcome, _ = run(world, [req], tmp_path / "m.json")
    assert not failed and not world.calls
    assert outcome.progress.counts()["chunks_reused"] == 1
    assert bar.return_value.set_postfix.call_args.kwargs["records_complete"] == 1
    assert bar.return_value.set_postfix.call_args.kwargs["chunks_reused"] == 1


def test_resume_before_due_time_defers_without_provider_work_then_retries_only_failure(tmp_path):
    now = [100.0]
    retries = RetryStore(tmp_path / "q", clock=lambda: now[0])
    requests = [make_request("r1", 0), make_request("r1", 1)]
    world = FakeWorld()
    world.fail_on = {requests[1].chunk_id}
    succeeded, failed, first = execute(world, requests, tmp_path, retries)
    assert len(succeeded) == len(failed) == 1
    world.fail_on.clear()
    world.calls.clear()
    now[0] = 104.99
    _, failed, second = execute(world, requests, tmp_path, retries)
    assert not failed and not world.calls
    assert second.progress.counts()["chunks_deferred"] == 1
    assert second.progress.counts()["records_partial"] == 1
    assert second.progress.to_dict()["retry_chunks"][1]["attempts"] == 1
    now[0] = 105
    _, failed, third = execute(world, requests, tmp_path, retries)
    assert not failed and world.calls == [requests[1].chunk_id]
    assert third.progress.counts()["chunks_retry_attempted"] == 1
    assert third.progress.counts()["records_complete"] == 1


def test_normal_resume_does_not_drain_failures_outside_selected_records(tmp_path):
    now = [100.0]
    retries = RetryStore(tmp_path / "q", clock=lambda: now[0])
    a, b = make_request("r1", 0), make_request("r2", 0)
    world = FakeWorld()
    world.fail_on = {b.chunk_id}
    execute(world, [a, b], tmp_path, retries)
    now[0] = 105
    world.fail_on.clear()
    world.calls.clear()
    _, _, outcome = execute(world, [a], tmp_path, retries)
    assert not world.calls
    assert retries.for_request(b).snapshot().chunks[b.chunk_id].status == "retry_wait"
    assert len(outcome.progress.retry_chunks) == 1


def test_exhaustion_is_visible_and_budget_not_reset_by_fresh_manifest(tmp_path):
    now = [100.0]
    retries = RetryStore(tmp_path / "q", clock=lambda: now[0])
    req = make_request("r1", 0)
    world = FakeWorld()
    world.fail_on = {req.chunk_id}
    for timestamp in [100, 105, 135]:
        now[0] = timestamp
        execute(world, [req], tmp_path, retries)
    row = retries.for_request(req).snapshot().chunks[req.chunk_id]
    assert row.status == "dead_letter" and row.terminal_reason == "exhausted"
    assert len(world.calls) == 3
    fresh = ManifestStore(state_dir=tmp_path / "m", fresh=True)
    retries.register([req], fresh)
    now[0] = 1000
    _, _, outcome = execute(world, [req], tmp_path, retries)
    assert len(world.calls) == 3
    assert outcome.progress.to_dict()["chunks_dead_lettered"] == 1
    assert outcome.progress.retry_chunks[0]["attempts"] == 3


def test_same_record_id_in_two_sources_has_independent_record_progress(tmp_path):
    requests = [make_request("r1", 0), make_request("r1", 0, sha="b" * 64)]
    # Fake graph keyed by source as well as chunk ID for this case.
    graph = set()
    pipeline = MagicMock()

    async def write(**kwargs):
        m = kwargs["document_metadata"]
        graph.add((m["source_sha256"], m["chunk_id"]))
        return MagicMock(run_id="ok")

    pipeline.run_async = write
    with patch.object(loader, "Neo4jConnection"), patch.object(loader, "build_simple_kg_pipeline", return_value=pipeline), patch.object(loader, "classify_chunk", side_effect=lambda _, r: ChunkState("complete" if (r.provenance.source_sha256, r.chunk_id) in graph else "absent")):
        outcome = loader.RunOutcome()
        asyncio.run(loader.run_requests(requests, MagicMock(), MagicMock(), show_progress=False, outcome=outcome))
    assert outcome.progress.counts()["records_complete"] == 2


def test_dry_run_reports_planned_records_as_unstarted_without_queue_or_services(tmp_path):
    requests = [make_request("r1", 0), make_request("r2", 0)]
    with patch.object(loader, "collect_requests", return_value=(requests, [], 2)), patch.object(loader, "Neo4jConnection", side_effect=AssertionError("dry run")), patch.object(loader, "RetryStore", side_effect=AssertionError("no local retry files")):
        summary = asyncio.run(loader.load_dataset(tmp_path / "missing.jsonl", 1, 2, "fixture", True))
    report = summary.to_dict()
    assert report["records_unstarted"] == report["records_planned"] == 2
    assert report["chunks_processed"] == 0 and report["queue_paths"] == []
    assert summary.succeeded
