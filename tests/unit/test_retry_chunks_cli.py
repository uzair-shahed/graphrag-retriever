"""Queue CLI behavior, bounded selection, and dataset-independent orchestration."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphrag_pipeline.ingestion import retry_chunks as cli
from graphrag_pipeline.ingestion.progress_manifest import ProgressManifest
from graphrag_pipeline.ingestion.retry_policy import FailureDecision
from graphrag_pipeline.ingestion.retry_queue import RetryQueue, QueueError
from test_load_resume import FakeWorld, make_request
from test_retry_capture import MODELS


@pytest.fixture
def queue(tmp_path):
    now = [100.0]
    queue = RetryQueue.open(tmp_path / "queue.json", "a" * 64, "fixture", clock=lambda: now[0])
    queue.register([make_request("r1", 0), make_request("r1", 1), make_request("r2", 0)])
    return queue, now


def drain(queue, selected, world, **kwargs):
    with patch.object(cli, "Neo4jConnection"), patch.object(cli, "build_simple_kg_pipeline", return_value=world.pipeline()), patch("graphrag_pipeline.extraction.ingest.classify_chunk", side_effect=world.classify):
        return asyncio.run(cli.drain_queue(queue, selected, neo4j_settings=MagicMock(), model_settings=MODELS, **kwargs))


def test_inspection_is_offline_and_does_not_mutate(queue, capsys):
    q, _ = queue
    before = q.path.read_bytes()
    with patch.object(cli, "Neo4jConnection", side_effect=AssertionError("must remain offline")), patch.object(cli, "model_settings_from_env", side_effect=AssertionError("no model settings")):
        assert cli.run_cli(["inspect", "--queue-path", str(q.path), "--record-id", "r1"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert len(report["entries"]) == 2
    assert report["authority"] == "local_queue_state; Neo4j_not_checked"
    assert q.path.read_bytes() == before


@pytest.mark.parametrize("filters", [(["missing"], []), ([], ["missing"]), (["r1"], ["musique:r2:paragraph:0"])])
def test_invalid_selection_is_rejected(queue, filters):
    q, _ = queue
    with pytest.raises(QueueError):
        cli.select_ids(q, *filters)


def test_requeue_is_atomic_for_mixed_terminal_and_nonterminal_targets(queue, capsys):
    q, _ = queue
    cid = "musique:r1:paragraph:0"
    q.record_observation(cid, FailureDecision("authentication", False), "extraction")
    before = q.path.read_bytes()
    assert cli.run_cli(["requeue", "--queue-path", str(q.path), "--chunk-id", cid, "--chunk-id", "musique:r1:paragraph:1", "--reason", "credentials fixed"]) == 2
    assert q.path.read_bytes() == before
    capsys.readouterr()
    assert cli.run_cli(["requeue", "--queue-path", str(q.path), "--chunk-id", cid, "--reason", "credentials fixed"]) == 0
    entry = RetryQueue.load(q.path).snapshot().chunks[cid]
    assert entry.status == "pending" and len(entry.cycles) == 2
    assert entry.cycles[-1].reason == "credentials fixed"


def test_requeue_requires_explicit_ids_and_nonblank_reason(queue, capsys):
    q, _ = queue
    with pytest.raises(SystemExit) as error:
        cli.run_cli(["requeue", "--queue-path", str(q.path), "--reason", "fixed"])
    assert error.value.code == 2
    before = q.path.read_bytes()
    assert cli.run_cli(["requeue", "--queue-path", str(q.path), "--chunk-id", "musique:r1:paragraph:0", "--reason", " "]) == 2
    assert q.path.read_bytes() == before


def test_chunk_filter_never_runs_missing_sibling_but_record_remains_partial(queue):
    q, _ = queue
    world = FakeWorld()
    selected = ["musique:r1:paragraph:0"]
    with patch("graphrag_pipeline.ingestion.load_musique_dataset.collect_requests", side_effect=AssertionError("no dataset scanning")):
        report = drain(q, selected, world)
    assert world.calls == selected and report["succeeded"]
    assert report["records_complete"] == 0 and report["records_partial"] == 1
    assert report["record_verification"][0]["verified_chunks"] == 1
    assert q.snapshot().chunks["musique:r1:paragraph:1"].status == "pending"


def test_bounded_drain_processes_one_candidate_and_reports_remaining_work(queue):
    q, _ = queue
    world = FakeWorld()
    selected = cli.select_ids(q, ["r1"])
    report = drain(q, selected, world, max_chunks=1)
    assert world.calls == selected[:1] and not report["succeeded"]
    assert report["chunks_unresolved"] == 1 and report["chunks_processed"] == 1


def test_drain_does_not_pick_up_newly_due_entries_mid_invocation(queue):
    q, now = queue
    a, b = cli.select_ids(q, ["r1"])
    q.begin_attempt(b, {})
    q.fail_attempt(b, FailureDecision("timeout", True), "extraction")
    world = FakeWorld()
    pipeline = world.pipeline()
    run = pipeline.run_async

    async def advancing(**kwargs):
        now[0] = 200
        return await run(**kwargs)

    pipeline.run_async = advancing
    with patch.object(cli, "Neo4jConnection"), patch.object(cli, "build_simple_kg_pipeline", return_value=pipeline), patch("graphrag_pipeline.extraction.ingest.classify_chunk", side_effect=world.classify):
        report = asyncio.run(cli.drain_queue(q, [a, b], neo4j_settings=MagicMock(), model_settings=MODELS))
    assert world.calls == [a] and report["chunks_deferred"] == 1
    assert q.snapshot().chunks[b].cycle_attempts == 1


def test_failed_attempt_keeps_backoff_and_does_not_retry_again_in_drain(queue):
    q, _ = queue
    world = FakeWorld()
    cid = "musique:r1:paragraph:0"
    world.fail_on = {cid}
    report = drain(q, [cid], world)
    assert world.calls == [cid] and report["chunks_failed_this_run"] == 1
    assert report["retry_chunks"][0]["next_attempt_at"] == 105
    world.calls.clear()
    report = drain(q, [cid], world)
    assert not world.calls and report["chunks_deferred"] == 1
    assert q.snapshot().chunks[cid].cycle_attempts == 1


def test_drain_updates_custom_linked_manifest_without_dataset(queue, tmp_path):
    q, _ = queue
    data = q.snapshot()
    reqs = [e.request for e in data.chunks.values() if e.request.record_id == "r1"]
    path = tmp_path / "custom-progress.json"
    manifest = ProgressManifest.open(path, "a" * 64, "fixture")
    manifest.register(reqs)
    q.remember_manifest(path)
    report = drain(q, cli.select_ids(q, ["r1"]), FakeWorld())
    after = ProgressManifest.open(path, "a" * 64, "fixture")
    assert report["records_complete"] == 1 and after.is_record_complete("r1")
    assert all(after.chunk_snapshot(r.chunk_id)["attempts"] == 1 for r in reqs)


def test_stale_verified_manifest_is_downgraded_when_sibling_is_missing(queue, tmp_path):
    q, _ = queue
    a, b = cli.select_ids(q, ["r1"])
    reqs = [q.snapshot().chunks[cid].request for cid in [a, b]]
    path = tmp_path / "m.json"
    m = ProgressManifest.open(path, "a" * 64, "fixture")
    m.register(reqs)
    m.mark_verified(b)
    q.remember_manifest(path)
    report = drain(q, [a], FakeWorld())
    assert report["records_partial"] == 1
    assert not ProgressManifest.open(path, "a" * 64, "fixture").is_record_complete("r1")


def test_mismatched_manifest_inventory_fails_before_model_work(queue, tmp_path):
    q, _ = queue
    cid = "musique:r1:paragraph:0"
    path = tmp_path / "m.json"
    m = ProgressManifest.open(path, "a" * 64, "fixture")
    m.register([q.snapshot().chunks[cid].request])
    q.remember_manifest(path)
    world = FakeWorld()
    with pytest.raises(cli.ManifestError):
        drain(q, [cid], world)
    assert not world.calls


def test_deleted_manifest_is_reported_and_does_not_block_source_independent_drain(queue, tmp_path):
    q, _ = queue
    missing = tmp_path / "deleted.json"
    q.remember_manifest(missing)
    report = drain(q, cli.select_ids(q, ["r1"]), FakeWorld())
    assert report["succeeded"] and report["missing_manifest_paths"] == [str(missing)]


def test_already_committed_work_needs_no_model_configuration(queue):
    q, _ = queue
    cid = "musique:r1:paragraph:0"
    world = FakeWorld()
    world.graph[cid] = "complete"
    with patch.object(cli, "model_settings_from_env", side_effect=AssertionError("no model settings needed")):
        report = drain(q, [cid], world)
    assert report["succeeded"] and not world.calls


def test_clean_stop_starts_no_new_provider_work(queue):
    q, _ = queue
    world = FakeWorld()
    report = drain(q, cli.select_ids(q), world, stop_requested=lambda: True)
    assert report["stopped_early"] and not world.calls
    assert all(not entry.history for entry in q.snapshot().chunks.values())


@pytest.mark.parametrize("outcome,code", [("success", 0), ("partial", 1), ("stop", 130)])
def test_cli_drain_exit_codes(queue, capsys, outcome, code):
    q, _ = queue
    result = {"succeeded": outcome == "success", "stopped_early": outcome == "stop"}
    with patch.object(cli, "drain_queue", new=AsyncMock(return_value=result)), patch.object(cli.signal, "signal", return_value=signal_handler):
        assert cli.run_cli(["drain", "--queue-path", str(q.path)]) == code
    assert json.loads(capsys.readouterr().out) == result


def signal_handler(*_args):
    pass


def test_missing_or_corrupt_queue_is_error_without_reset(tmp_path, capsys):
    path = tmp_path / "q.json"
    assert cli.run_cli(["inspect", "--queue-path", str(path)]) == 2
    assert not path.exists()
    path.write_text("{broken")
    assert cli.run_cli(["inspect", "--queue-path", str(path)]) == 2
    assert path.read_text() == "{broken"


def test_invalid_max_chunks_is_parser_error(queue):
    q, _ = queue
    with pytest.raises(SystemExit) as error:
        cli.run_cli(["drain", "--queue-path", str(q.path), "--max-chunks", "0"])
    assert error.value.code == 2


def test_model_configuration_is_captured_when_pipeline_is_built_lazily(queue):
    q, _ = queue
    cid = "musique:r1:paragraph:0"
    report = drain(q, [cid], FakeWorld())
    assert report["succeeded"]
    config = q.snapshot().chunks[cid].history[0].configuration
    assert config["provider"] == "ollama" and config["llm_model"] == "fixed"
    assert config["embedding_model"] == "fixed" and config["ingestion_contract_version"] == 1


def test_requeue_then_failed_drain_preserves_new_cycle_backoff(queue):
    q, now = queue
    cid = "musique:r1:paragraph:0"
    q.record_observation(cid, FailureDecision("authentication", False), "extraction")
    q.requeue(cid, "credentials repaired")
    world = FakeWorld()
    world.fail_on = {cid}
    report = drain(q, [cid], world)
    assert report["chunks_deferred"] == 1
    assert q.snapshot().chunks[cid].status == "retry_wait"
    assert q.snapshot().chunks[cid].next_attempt_at == now[0] + 5
    assert len(q.snapshot().chunks[cid].cycles) == 2


def test_corrupt_manifest_source_stops_before_provider_calls(queue, tmp_path):
    q, _ = queue
    path = tmp_path / "wrong-source.json"
    m = ProgressManifest.open(path, "b" * 64, "fixture")
    m.flush()
    q.remember_manifest(path)
    world = FakeWorld()
    with pytest.raises(cli.ManifestError):
        drain(q, cli.select_ids(q), world)
    assert not world.calls
