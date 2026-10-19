"""Failure persistence and cross-journal recovery at the shared boundary."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphrag_pipeline.config import OllamaSettings
from graphrag_pipeline.extraction.ingest import ingest_paragraph, RetryDeferred
from graphrag_pipeline.extraction.run_fixture import run_fixture
from graphrag_pipeline.ingestion import load_musique_dataset as loader
from graphrag_pipeline.ingestion.progress_manifest import ManifestError, ManifestStore, ProgressManifest
from graphrag_pipeline.ingestion.reconcile import ABSENT, COMPLETE, PARTIAL, CONFLICT, LEGACY, ChunkState
from graphrag_pipeline.ingestion.retry_policy import FailureDecision, PipelineStageError
from graphrag_pipeline.ingestion.retry_queue import RetryStore, RetryQueue, QueueError
from test_load_resume import make_request, four_chunks


MODELS = OllamaSettings(host="http://unused", llm_model="fixed", embedding_model="fixed")


def entry(req):
    return RetryStore().for_request(req).snapshot().chunks[req.chunk_id]


def test_failure_is_durable_before_loader_continues_to_sibling(tmp_path):
    requests = four_chunks()
    state = {}
    calls = []

    async def execute(**kwargs):
        cid = kwargs["document_metadata"]["chunk_id"]
        calls.append(cid)
        if cid == requests[0].chunk_id:
            raise TimeoutError("model unavailable")
        assert entry(requests[0]).status == "retry_wait"
        assert entry(requests[0]).history[0].category == "timeout"
        state[cid] = COMPLETE
        return SimpleNamespace(run_id="ok")

    pipeline = SimpleNamespace(run_async=execute)
    store = ManifestStore(state_dir=tmp_path / "manifests")
    with patch.object(loader, "Neo4jConnection"), patch.object(loader, "build_simple_kg_pipeline", return_value=pipeline), patch.object(loader, "classify_chunk", side_effect=lambda _, r: ChunkState(state.get(r.chunk_id, ABSENT))):
        succeeded, failed = asyncio.run(loader.run_requests(requests, MagicMock(), MODELS, store=store, show_progress=False))
    assert len(succeeded) == 3 and len(failed) == 1 and calls == [r.chunk_id for r in requests]
    assert store.complete_record_ids(requests) == {"r2"}
    assert len(entry(requests[0]).history) == 1
    assert entry(requests[1]).status == "resolved"


@pytest.mark.parametrize("stage", ["begin", "failure", "resolution"])
def test_queue_persistence_error_stops_batch_before_next_chunk(tmp_path, stage):
    requests = four_chunks()
    pipeline = MagicMock()
    pipeline.run_async = AsyncMock(side_effect=TimeoutError("fail")) if stage == "failure" else AsyncMock(return_value=SimpleNamespace(run_id="ok"))
    method = {"begin": "begin_attempt", "failure": "fail_attempt", "resolution": "resolve_verified"}[stage]
    states = [ChunkState(ABSENT), ChunkState(COMPLETE)]
    with patch.object(loader, "Neo4jConnection"), patch.object(loader, "build_simple_kg_pipeline", return_value=pipeline), patch.object(loader, "classify_chunk", side_effect=states), patch.object(RetryQueue, method, side_effect=QueueError("disk unavailable")):
        with pytest.raises(QueueError, match="disk unavailable"):
            asyncio.run(loader.run_requests(requests, MagicMock(), MODELS, store=ManifestStore(state_dir=tmp_path / "m"), show_progress=False))
    assert pipeline.run_async.await_count == (0 if stage == "begin" else 1)
    assert entry(requests[1]).status == "pending"


def test_diagnostic_failure_cannot_hide_original_queue_failure(tmp_path):
    req = make_request("r1", 0)
    pipeline = MagicMock()
    pipeline.run_async = AsyncMock(side_effect=TimeoutError("original failure"))
    with patch.object(loader, "Neo4jConnection"), patch.object(loader, "build_simple_kg_pipeline", return_value=pipeline), patch.object(loader, "classify_chunk", return_value=ChunkState(ABSENT)), patch.object(loader, "save_invalid_llm_response", side_effect=OSError("debug directory unwritable")):
        _, failed = asyncio.run(loader.run_requests([req], MagicMock(), MODELS, show_progress=False))
    assert failed[0]["error_type"] == "TimeoutError"
    assert failed[0]["diagnostic_error"]["error_type"] == "OSError"
    assert entry(req).last_error_category == "timeout"


@pytest.mark.parametrize("state", [PARTIAL, CONFLICT, LEGACY])
def test_unsafe_graph_state_is_dead_lettered_without_provider_calls(state):
    req = make_request("r1", 0)
    pipeline = MagicMock()
    result = asyncio.run(ingest_paragraph(None, pipeline, req, classify=lambda *_: ChunkState(state)))
    assert result.state.state == state
    assert entry(req).status == "dead_letter"
    assert entry(req).last_error_category == "unsafe_graph_state"
    assert not entry(req).history
    pipeline.run_async.assert_not_called()


def test_committed_graph_resolves_queue_before_manifest_checkpoint_and_recovers(tmp_path):
    req = make_request("r1", 0)
    graph = {}
    pipeline = MagicMock()

    async def write(**kwargs):
        graph[req.chunk_id] = COMPLETE
        return SimpleNamespace(run_id="committed")

    pipeline.run_async = AsyncMock(side_effect=write)
    path = tmp_path / "manifest.json"
    store = ManifestStore(manifest_path=path)
    actual = ProgressManifest.mark_verified

    def crash_checkpoint(self, cid):
        assert entry(req).status == "resolved"
        raise ManifestError("checkpoint unavailable")

    with patch.object(loader, "Neo4jConnection"), patch.object(loader, "build_simple_kg_pipeline", return_value=pipeline), patch.object(loader, "classify_chunk", side_effect=lambda *_: ChunkState(graph.get(req.chunk_id, ABSENT))):
        with patch.object(ProgressManifest, "mark_verified", crash_checkpoint), pytest.raises(ManifestError):
            asyncio.run(loader.run_requests([req], MagicMock(), MODELS, store=store, show_progress=False))
        assert json.loads(path.read_text())["chunks"][req.chunk_id]["status"] == "in_flight"
        outcome = loader.RunOutcome()
        store = ManifestStore(manifest_path=path)
        with patch.object(ProgressManifest, "mark_verified", actual):
            succeeded, failed = asyncio.run(loader.run_requests([req], MagicMock(), MODELS, store=store, outcome=outcome, show_progress=False))
    assert pipeline.run_async.await_count == 1
    assert not succeeded and not failed and outcome.skipped_verified == [req.chunk_id]
    assert store.complete_record_ids([req]) == {req.record_id}


def test_interrupted_attempt_is_reconciled_before_replay(tmp_path):
    req = make_request("r1", 0)
    now = [100.0]
    queue = RetryQueue.open(tmp_path / "q.json", "a" * 64, "fixture", clock=lambda: now[0])
    queue.register([req])
    queue.begin_attempt(req.chunk_id, {})
    checks = []
    pipeline = MagicMock()

    def classify(*_):
        checks.append("read")
        return ChunkState(ABSENT)

    with pytest.raises(RetryDeferred):
        asyncio.run(ingest_paragraph(None, pipeline, req, retry_queue=queue, classify=classify))
    assert checks == ["read"] and queue.snapshot().chunks[req.chunk_id].status == "retry_wait"
    assert len(queue.snapshot().chunks[req.chunk_id].history) == 1
    pipeline.run_async.assert_not_called()


def test_reconciliation_failure_is_saved_without_spending_provider_attempt():
    req = make_request("r1", 0)

    def fail(*_):
        raise TimeoutError("read failed")

    with pytest.raises(TimeoutError):
        asyncio.run(ingest_paragraph(None, MagicMock(), req, classify=fail))
    assert entry(req).status == "retry_wait"
    assert entry(req).observations[0].stage == "reconcile"
    assert not entry(req).history


def test_pipeline_construction_failure_is_durable_configuration_dead_letter():
    req = make_request("r1", 0)
    factory = MagicMock(side_effect=ValueError("bad model settings"))
    with pytest.raises(ValueError):
        asyncio.run(ingest_paragraph(None, None, req, classify=lambda *_: ChunkState(ABSENT), pipeline_factory=factory))
    assert entry(req).status == "dead_letter"
    assert entry(req).last_error_category == "configuration"
    assert not entry(req).history


@pytest.mark.parametrize("stage", ["graph_validation", "embedding", "write"])
def test_pipeline_failure_retains_typed_stage(stage):
    req = make_request("r1", 0)
    error = PipelineStageError(stage, "component failed")
    error.__cause__ = ValueError("bad generated graph") if stage == "graph_validation" else TimeoutError()
    pipeline = SimpleNamespace(run_async=AsyncMock(side_effect=error))
    with pytest.raises(PipelineStageError):
        asyncio.run(ingest_paragraph(None, pipeline, req, classify=lambda *_: ChunkState(ABSENT)))
    assert entry(req).history[0].stage == stage
    assert entry(req).status == "retry_wait"


@pytest.mark.parametrize("attempts", [0, 2, 7])
def test_legacy_failure_import_preserves_attempts_and_original_manifest(tmp_path, attempts):
    req = make_request("r1", 0)
    store = ManifestStore(state_dir=tmp_path / "m")
    store.register([req])
    manifest = store.for_request(req)
    for _ in range(attempts):
        manifest.mark_in_flight(req.chunk_id)
    manifest.mark_failed(req.chunk_id, {"error_type": "RuntimeError", "error": "old failure"})
    before = manifest.path.read_bytes()
    retries = RetryStore()
    retries.register([req], store)
    saved = entry(req)
    assert saved.legacy_attempts == attempts and saved.status == "dead_letter"
    assert saved.last_error_category == "legacy_unclassified"
    assert manifest.path.read_bytes() == before
    # Migration is idempotent; requeue preserves lifetime attempt numbering.
    retries.register([req], store)
    queue = retries.for_request(req)
    queue.requeue(req.chunk_id, "reviewed source failure")
    queue.begin_attempt(req.chunk_id, {})
    assert queue.snapshot().chunks[req.chunk_id].history[0].number == attempts + 1


def test_fixture_failure_is_queued_even_if_diagnostic_logging_fails(tmp_path):
    req = make_request("fixture-record", 0)
    path = tmp_path / "fixture.json"
    path.write_text(req.model_dump_json())
    pipeline = MagicMock()
    pipeline.run_async = AsyncMock(side_effect=TimeoutError("fixture provider down"))
    with patch("graphrag_pipeline.extraction.run_fixture.Neo4jSettings.from_env", return_value=MagicMock()), patch("graphrag_pipeline.extraction.run_fixture.model_settings_from_env", return_value=MODELS), patch("graphrag_pipeline.extraction.run_fixture.Neo4jConnection"), patch("graphrag_pipeline.extraction.run_fixture.build_simple_kg_pipeline", return_value=pipeline), patch("graphrag_pipeline.extraction.ingest.classify_chunk", return_value=ChunkState(ABSENT)), patch("graphrag_pipeline.extraction.run_fixture.save_invalid_llm_response", side_effect=OSError("debug failed")):
        with pytest.raises(TimeoutError, match="fixture provider down"):
            asyncio.run(run_fixture(path))
    assert entry(req).status == "retry_wait"


def test_fresh_manifest_imports_old_failure_without_resetting_attempts(tmp_path):
    req = make_request("r1", 0)
    path = tmp_path / "manifest.json"
    original = ManifestStore(manifest_path=path)
    original.register([req])
    original.for_request(req).mark_in_flight(req.chunk_id)
    original.for_request(req).mark_failed(req.chunk_id, {"error": "old unknown failure"})
    before = path.read_bytes()
    fresh = ManifestStore(manifest_path=path, fresh=True)
    fresh.register([req])
    RetryStore().register([req], fresh)
    assert entry(req).legacy_attempts == 1 and entry(req).status == "dead_letter"
    assert list(tmp_path.glob("manifest.json.superseded-*"))[0].read_bytes() == before


def test_transient_reconciliation_error_does_not_reactivate_dead_letter():
    req = make_request("r1", 0)
    queue = RetryStore().for_request(req)
    queue.ensure_request(req)
    queue.record_observation(req.chunk_id, FailureDecision("unsafe_graph_state", False), "reconcile")

    def unreadable(*_):
        raise TimeoutError("cannot inspect database")

    with pytest.raises(TimeoutError):
        asyncio.run(ingest_paragraph(None, MagicMock(), req, retry_queue=queue, classify=unreadable))
    assert entry(req).status == "dead_letter" and entry(req).last_error_category == "timeout"


def test_preload_failure_uses_shared_capture_and_keeps_record_partial(tmp_path):
    reqs = [make_request("r1", 0), make_request("r1", 1)]
    state = {}

    async def execute(**kwargs):
        cid = kwargs["document_metadata"]["chunk_id"]
        if cid == reqs[0].chunk_id:
            raise TimeoutError()
        state[cid] = COMPLETE
        return SimpleNamespace(run_id="ok")

    with patch.object(loader, "collect_selected_requests", return_value=(reqs, [], 1)), patch.object(loader, "Neo4jConnection"), patch.object(loader, "build_simple_kg_pipeline", return_value=SimpleNamespace(run_async=execute)), patch.object(loader, "classify_chunk", side_effect=lambda _, r: ChunkState(state.get(r.chunk_id, ABSENT))):
        summary = asyncio.run(loader.load_selected_records([], neo4j_settings=MagicMock(), openrouter_settings=MODELS, state_dir=tmp_path / "m", show_progress=False))
    assert not summary.succeeded and summary.records_succeeded == 0
    assert entry(reqs[0]).status == "retry_wait" and entry(reqs[1]).status == "resolved"


@pytest.mark.parametrize("command", ["batch", "preload", "fixture"])
def test_runtime_configuration_failures_are_durable_before_return(tmp_path, command):
    req = make_request("r1", 0)
    failure = ValueError("missing provider configuration")
    with patch.object(loader.Neo4jSettings, "from_env", return_value=MagicMock()), patch.object(loader, "model_settings_from_env", side_effect=failure):
        if command == "batch":
            with patch.object(loader, "collect_requests", return_value=([req], [], 1)), pytest.raises(ValueError):
                asyncio.run(loader.load_dataset(tmp_path / "dataset.jsonl", 1, 1, "fixture", False, state_dir=tmp_path / "m"))
        elif command == "preload":
            with patch.object(loader, "collect_selected_requests", return_value=([req], [], 1)), pytest.raises(ValueError):
                asyncio.run(loader.load_selected_records([], state_dir=tmp_path / "m"))
        else:
            path = tmp_path / "fixture.json"
            path.write_text(req.model_dump_json())
            with patch("graphrag_pipeline.extraction.run_fixture.model_settings_from_env", side_effect=failure), pytest.raises(ValueError):
                asyncio.run(run_fixture(path))
    assert entry(req).status == "dead_letter"
    assert entry(req).last_error_category == "configuration" and not entry(req).history


def test_connection_creation_failure_is_recorded_for_all_selected_chunks():
    reqs = four_chunks()
    connection = MagicMock()
    connection.connect.side_effect = ConnectionError("driver setup failed")
    with patch.object(loader, "Neo4jConnection", return_value=connection), pytest.raises(ConnectionError):
        asyncio.run(loader.run_requests(reqs, MagicMock(), MODELS, show_progress=False))
    assert all(entry(req).status == "retry_wait" and not entry(req).history for req in reqs)
    connection.close.assert_called_once()


def test_batch_without_manifest_still_captures_provider_failure():
    req = make_request("r1", 0)
    pipeline = SimpleNamespace(run_async=AsyncMock(side_effect=TimeoutError()))
    with patch.object(loader, "Neo4jConnection"), patch.object(loader, "build_simple_kg_pipeline", return_value=pipeline), patch.object(loader, "classify_chunk", return_value=ChunkState(ABSENT)):
        _, failed = asyncio.run(loader.run_requests([req], MagicMock(), MODELS, store=None, show_progress=False))
    assert failed and entry(req).status == "retry_wait"


def test_saved_configuration_excludes_credentials():
    req = make_request("r1", 0)
    pipeline = SimpleNamespace(run_async=AsyncMock(side_effect=TimeoutError()))
    model = SimpleNamespace(llm_model="fixed", embedding_model="fixed", api_key="never-persist-this", password="never-persist-this")
    settings = SimpleNamespace(database="neo4j", password="never-persist-this")
    with patch.object(loader, "Neo4jConnection"), patch.object(loader, "build_simple_kg_pipeline", return_value=pipeline), patch.object(loader, "classify_chunk", return_value=ChunkState(ABSENT)):
        asyncio.run(loader.run_requests([req], settings, model, show_progress=False))
    assert entry(req).history[0].configuration["llm_model"] == "fixed"
    assert "never-persist-this" not in RetryStore().for_request(req).path.read_text()


def test_fresh_manifest_archiving_waits_for_durable_legacy_import(tmp_path):
    req = make_request("r1", 0)
    path = tmp_path / "manifest.json"
    old = ManifestStore(manifest_path=path)
    old.register([req])
    old.for_request(req).mark_in_flight(req.chunk_id)
    old.for_request(req).mark_failed(req.chunk_id, {"error": "legacy"})
    before = path.read_bytes()
    fresh = ManifestStore(manifest_path=path, fresh=True)
    with patch.object(RetryQueue, "import_legacy_failure", side_effect=QueueError("cannot save history")), pytest.raises(QueueError):
        RetryStore().register([req], fresh)
    assert path.read_bytes() == before and not list(tmp_path.glob("*.superseded-*"))
    RetryStore().register([req], fresh)
    assert entry(req).legacy_attempts == 1 and entry(req).status == "dead_letter"
    assert list(tmp_path.glob("*.superseded-*"))


def test_historical_synthetic_fixture_provenance_is_not_rewritten():
    from graphrag_pipeline.extraction.run_fixture import load_fixture, DEFAULT_FIXTURE

    req = load_fixture(DEFAULT_FIXTURE)
    queue = RetryStore().for_request(req)
    queue.ensure_request(req)
    assert queue.path.name.startswith("opaque-")
    assert queue.snapshot().chunks[req.chunk_id].request == req
    assert req.provenance.source_sha256 == "fixture-sha256"


def test_pre_write_checkpoint_failure_does_not_consume_provider_attempt():
    req = make_request("r1", 0)
    pipeline = SimpleNamespace(run_async=AsyncMock())
    with pytest.raises(ManifestError):
        asyncio.run(ingest_paragraph(None, pipeline, req, classify=lambda *_: ChunkState(ABSENT), before_write=MagicMock(side_effect=OSError("disk error"))))
    assert not entry(req).history and entry(req).status == "pending"
    pipeline.run_async.assert_not_awaited()


def test_default_verifier_runs_before_and_after_successful_pipeline():
    req = make_request("r1", 0)
    pipeline = SimpleNamespace(run_async=AsyncMock(return_value=SimpleNamespace(run_id="ok")))
    with patch("graphrag_pipeline.extraction.ingest.classify_chunk", side_effect=[ChunkState(ABSENT), ChunkState(COMPLETE)]) as verifier:
        result = asyncio.run(ingest_paragraph(None, pipeline, req))
    assert result.state.state == COMPLETE and not result.reused
    assert verifier.call_count == 2
    assert entry(req).status == "resolved" and entry(req).history[0].outcome == "verified"


@pytest.mark.parametrize("attempts", [1, 3, 7])
def test_pre_queue_interrupted_manifest_preserves_budget_until_graph_reconciliation(tmp_path, attempts):
    req = make_request("r1", 0)
    manifest_path = tmp_path / "pre-queue.json"
    old = ManifestStore(manifest_path=manifest_path)
    old.register([req])
    for _ in range(attempts):
        old.for_request(req).mark_in_flight(req.chunk_id)
    now = [100.0]
    retries = RetryStore(tmp_path / "q", clock=lambda: now[0])
    resumed = ManifestStore(manifest_path=manifest_path)
    retries.register([req], resumed)
    queue = retries.for_request(req)
    imported = queue.snapshot().chunks[req.chunk_id]
    assert imported.legacy_attempts == attempts and not imported.history
    assert imported.status == ("retry_wait" if attempts < 3 else "dead_letter")
    pipeline = SimpleNamespace(run_async=AsyncMock())
    # Even exhausted pre-queue attempts can reconcile a matching actual commit.
    result = asyncio.run(ingest_paragraph(None, pipeline, req, retry_queue=queue, classify=lambda *_: ChunkState(COMPLETE)))
    assert result.reused and queue.snapshot().chunks[req.chunk_id].status == "resolved"
    assert queue.snapshot().chunks[req.chunk_id].legacy_attempts == attempts
    pipeline.run_async.assert_not_awaited()


def test_pre_queue_interrupted_absent_work_does_not_get_a_fresh_budget(tmp_path):
    req = make_request("r1", 0)
    old = ManifestStore(manifest_path=tmp_path / "m.json")
    old.register([req])
    for _ in range(3):
        old.for_request(req).mark_in_flight(req.chunk_id)
    retries = RetryStore(tmp_path / "q")
    retries.register([req], ManifestStore(manifest_path=tmp_path / "m.json", fresh=True))
    queue = retries.for_request(req)
    pipeline = SimpleNamespace(run_async=AsyncMock())
    with pytest.raises(RetryDeferred):
        asyncio.run(ingest_paragraph(None, pipeline, req, retry_queue=queue, classify=lambda *_: ChunkState(ABSENT)))
    assert queue.snapshot().chunks[req.chunk_id].terminal_reason == "exhausted"
    pipeline.run_async.assert_not_awaited()
