"""Task 12.2 acceptance: real Neo4j/pipeline, deterministic providers only.

Opt in with RUN_TASK12_RETRY_TESTS=1. Missing Neo4j fails once enabled.
The shared Task 13 fixture audits cleanup against complete graph snapshot hashes.
"""

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from neo4j_graphrag.llm.types import LLMResponse

from graphrag_pipeline.config import OllamaSettings
from graphrag_pipeline.extraction.atomic_writer import AtomicKGWriter
from graphrag_pipeline.extraction.idempotency import read_inventory
from graphrag_pipeline.extraction.ingest import ingest_paragraph
from graphrag_pipeline.extraction.plugin_input import build_plugin_input
from graphrag_pipeline.extraction.run_fixture import run_fixture
from graphrag_pipeline.ingestion import load_musique_dataset as loader
from graphrag_pipeline.ingestion.progress_manifest import ManifestError, ManifestStore, ProgressManifest
from graphrag_pipeline.ingestion.reconcile import ABSENT, COMPLETE, PARTIAL, LEGACY, CONFLICT, classify_chunk
from graphrag_pipeline.ingestion.retry_queue import QueueError, RetryQueue, RetryStore
from graphrag_pipeline.query import select_queries
from test_atomic_ingestion import world as atomic_world
from test_stock_writer_behavior import FixedGraphLLM, FixedEmbedder, _snapshot


MODELS = OllamaSettings(host="http://unused", llm_model="fixed", embedding_model="fixed")


@pytest.fixture
def world(tmp_path, request, monkeypatch):
    if os.environ.get("RUN_TASK12_RETRY_TESTS") != "1":
        pytest.skip("Opt in with RUN_TASK12_RETRY_TESTS=1; writes and cleans UUID-scoped fixtures")
    monkeypatch.setenv("RUN_TASK13_ATOMIC_TESTS", "1")
    monkeypatch.setenv("TASK13_EVIDENCE_DIR", os.environ.get("TASK12_EVIDENCE_DIR", str(tmp_path)))
    yield from atomic_world.__wrapped__(tmp_path, request, monkeypatch)


def saved(request):
    return RetryStore().for_request(request).snapshot().chunks[request.chunk_id]


def test_real_rollback_failure_is_queued_before_siblings_continue(world, tmp_path, monkeypatch):
    requests = [world.request(0), world.request(1)]
    pipeline, llm, embedder = world.pipeline()
    original = AtomicKGWriter._write

    def fail(self, connection, request, *args):
        result = original(self, connection, request, *args)
        if request.chunk_id == requests[0].chunk_id:
            raise TimeoutError("injected failure after graph and completion writes, before commit")
        assert saved(requests[0]).status == "retry_wait"
        return result

    monkeypatch.setattr(AtomicKGWriter, "_write", fail)
    store = ManifestStore(manifest_path=tmp_path / "manifest.json")
    with patch.object(loader, "build_simple_kg_pipeline", return_value=pipeline):
        succeeded, failed = asyncio.run(loader.run_requests(requests, world.settings, MODELS, store=store, show_progress=False))
    assert len(succeeded) == len(failed) == 1
    assert classify_chunk(world.connection, requests[0]).state == ABSENT
    assert read_inventory(world.connection, requests[0].chunk_id) == {"nodes": [], "relationships": []}
    assert classify_chunk(world.connection, requests[1]).state == COMPLETE
    assert store.complete_record_ids(requests) == set()
    assert saved(requests[0]).history[0].stage == "write"
    assert saved(requests[0]).last_error_category == "timeout"
    assert saved(requests[1]).status == "resolved"
    assert llm.calls == embedder.calls == 2
    world.evidence.update(queue=[saved(r).model_dump(mode="json") for r in requests], failed=failed,
                          record_incomplete=True, rollback_verified=True, provider_calls=llm.calls)


def test_post_commit_verification_timeout_recovers_without_provider_replay(world):
    req = world.request()
    pipeline, llm, embedder = world.pipeline()
    checks = []

    def classifier(connection, request):
        checks.append(1)
        if len(checks) == 2:
            raise TimeoutError("verification connection lost after commit")
        return classify_chunk(connection, request)

    with pytest.raises(TimeoutError):
        asyncio.run(ingest_paragraph(world.connection, pipeline, req, classify=classifier))
    assert saved(req).status == "retry_wait" and saved(req).history[0].stage == "reconcile"
    assert classify_chunk(world.connection, req).state == COMPLETE
    before = read_inventory(world.connection, req.chunk_id)
    result = asyncio.run(ingest_paragraph(world.connection, pipeline, req))
    assert result.reused and saved(req).status == "resolved"
    assert llm.calls == embedder.calls == 1
    assert read_inventory(world.connection, req.chunk_id) == before
    world.evidence.update(queue=saved(req).model_dump(mode="json"), inventory=before,
                          provider_calls=llm.calls, committed_work_reused=True)


def test_queue_resolution_failure_leaves_recoverable_commit(world):
    req = world.request()
    pipeline, llm, embedder = world.pipeline()
    with patch.object(RetryQueue, "resolve_verified", side_effect=QueueError("injected local disk failure")), pytest.raises(QueueError):
        asyncio.run(ingest_paragraph(world.connection, pipeline, req))
    assert saved(req).status == "in_flight"
    assert classify_chunk(world.connection, req).state == COMPLETE
    before = read_inventory(world.connection, req.chunk_id)
    assert asyncio.run(ingest_paragraph(world.connection, pipeline, req)).reused
    assert saved(req).status == "resolved" and llm.calls == embedder.calls == 1
    assert read_inventory(world.connection, req.chunk_id) == before
    world.evidence.update(queue=saved(req).model_dump(mode="json"), inventory=before,
                          queue_write_failure_recovered=True, provider_calls=llm.calls)


def test_queue_resolves_before_manifest_and_restart_repairs_checkpoint(world, tmp_path):
    req = world.request()
    pipeline, llm, embedder = world.pipeline()
    path = tmp_path / "manifest.json"

    def fail_checkpoint(*_):
        assert saved(req).status == "resolved"
        raise ManifestError("injected manifest disk failure")

    with patch.object(loader, "build_simple_kg_pipeline", return_value=pipeline):
        with patch.object(ProgressManifest, "mark_verified", fail_checkpoint), pytest.raises(ManifestError):
            asyncio.run(loader.run_requests([req], world.settings, MODELS, store=ManifestStore(manifest_path=path), show_progress=False))
        checkpoint = json.loads(path.read_text())
        assert checkpoint["chunks"][req.chunk_id]["status"] == "in_flight"
        before = read_inventory(world.connection, req.chunk_id)
        resumed = ManifestStore(manifest_path=path)
        succeeded, failed = asyncio.run(loader.run_requests([req], world.settings, MODELS, store=resumed, show_progress=False))
    assert not succeeded and not failed and resumed.complete_record_ids([req]) == {req.record_id}
    assert llm.calls == embedder.calls == 1
    assert read_inventory(world.connection, req.chunk_id) == before
    world.evidence.update(interrupted_manifest=checkpoint, queue=saved(req).model_dump(mode="json"),
                          manifest=json.loads(path.read_text()), provider_calls=llm.calls, inventory=before)


@pytest.mark.parametrize("kind", [PARTIAL, LEGACY, CONFLICT])
def test_unsafe_existing_graph_is_preserved_and_dead_lettered(world, kind):
    req = world.request()
    _, metadata = build_plugin_input(req)
    if kind == PARTIAL:
        world.connection.run_write("CREATE (:Chunk {chunk_id: $id, text: $text})", {"id": req.chunk_id, "text": req.text})
    else:
        if kind == CONFLICT:
            metadata["record_id"] = "conflicting-record"
        world.connection.run_write("CREATE (d:Document) SET d = $props CREATE (c:Chunk) SET c = $props SET c.text = $text CREATE (c)-[:FROM_DOCUMENT]->(d)", {"props": metadata, "id": req.chunk_id, "text": req.text})
    assert classify_chunk(world.connection, req).state == kind
    before = _snapshot(world.connection)
    pipeline, llm, embedder = world.pipeline()
    result = asyncio.run(ingest_paragraph(world.connection, pipeline, req))
    assert result.state.state == kind
    assert saved(req).status == "dead_letter" and saved(req).last_error_category == "unsafe_graph_state"
    assert llm.calls == embedder.calls == 0 and _snapshot(world.connection) == before
    world.evidence.update(state=kind, queue=saved(req).model_dump(mode="json"), existing_data_preserved=True, provider_calls=0)


@pytest.mark.parametrize("command", ["preload", "fixture"])
def test_production_entrypoints_capture_model_failure(world, tmp_path, command):
    dataset = tmp_path / "musique_full_v1.0_dev.jsonl"
    dataset.write_text(json.dumps({"id": world.run_id, "question": "Who founded Aster?", "paragraphs": [{"idx": 0, "title": "Aster", "paragraph_text": "Alex founded Aster."}]}) + "\n")
    requests, _, _ = loader.collect_requests(dataset, 1, 1, "full_dev")
    world.requests.extend(requests)
    req = requests[0]
    llm, embedder = FixedGraphLLM(world.run_id), FixedEmbedder()

    def fail(*args, **kwargs):
        llm.calls += 1
        raise TimeoutError("injected provider outage")

    llm.invoke = fail
    with patch("graphrag_pipeline.extraction.pipeline.OllamaJSONSchemaLLM", return_value=llm), patch("graphrag_pipeline.extraction.pipeline.OllamaEmbeddings", return_value=embedder), patch("graphrag_pipeline.extraction.run_fixture.model_settings_from_env", return_value=MODELS):
        if command == "preload":
            selected = select_queries([dataset], record_id=world.run_id, preload=True)
            summary = asyncio.run(loader.load_selected_records(selected, neo4j_settings=world.settings, openrouter_settings=MODELS, show_progress=False, state_dir=tmp_path / "m"))
            assert not summary.succeeded and summary.records_succeeded == 0
            world.evidence["summary"] = summary.to_dict()
        else:
            fixture = tmp_path / "fixture.json"
            fixture.write_text(req.model_dump_json())
            with pytest.raises(Exception, match="injected provider outage"):
                asyncio.run(run_fixture(fixture))
    assert saved(req).status == "retry_wait" and saved(req).last_error_category == "timeout"
    assert classify_chunk(world.connection, req).state == ABSENT
    assert llm.calls == 1
    world.evidence.update(command=command, queue=saved(req).model_dump(mode="json"), provider_calls=llm.calls)


def test_malformed_model_output_is_captured_through_real_pipeline(world):
    req = world.request()
    pipeline, llm, embedder = world.pipeline()

    def malformed(*args, **kwargs):
        llm.calls += 1
        return LLMResponse(content='{"nodes": "wrong type", "relationships": []}')

    llm.invoke = malformed
    with pytest.raises(Exception):
        asyncio.run(ingest_paragraph(world.connection, pipeline, req))
    assert saved(req).status == "retry_wait"
    assert saved(req).last_error_category == "malformed_model_output"
    assert classify_chunk(world.connection, req).state == ABSENT
    assert llm.calls == 1
    world.evidence.update(queue=saved(req).model_dump(mode="json"), provider_calls=llm.calls, malformed_output_queued=True)


def test_failed_chunk_resume_in_new_process_skips_successful_sibling(world, tmp_path):
    import subprocess
    import sys
    from collections import Counter

    requests = [world.request(0), world.request(1)]
    request_path, manifest, events = [tmp_path / name for name in ("requests.json", "manifest.json", "events.jsonl")]
    request_path.write_text(json.dumps([r.model_dump(mode="json") for r in requests]))
    command = [sys.executable, str(Path(__file__).with_name("retry_ingestion_child.py")), str(request_path), str(manifest), str(events)]
    first = subprocess.run([*command, "--target", requests[1].chunk_id, "--fail"], capture_output=True, text=True, timeout=45)
    assert first.returncode == 1, first.stderr
    report = json.loads(first.stdout)
    assert report["outcome"]["progress"]["records_partial"] == 1
    assert report["outcome"]["progress"]["chunks_failed_this_run"] == 1
    before = read_inventory(world.connection, requests[0].chunk_id)
    assert saved(requests[1]).status == "retry_wait"
    second = subprocess.run(command, capture_output=True, text=True, timeout=45)
    assert second.returncode == 0, second.stderr
    resumed = json.loads(second.stdout)
    progress = resumed["outcome"]["progress"]
    assert progress["records_complete"] == 1 and progress["chunks_reused"] == 1
    assert progress["chunks_retry_attempted"] == 1 and progress["chunks_newly_verified"] == 1
    calls = Counter((row["chunk_id"], row["kind"]) for row in map(json.loads, events.read_text().splitlines()))
    assert calls == Counter({(requests[0].chunk_id, "llm"): 1, (requests[0].chunk_id, "embedding"): 1,
                             (requests[1].chunk_id, "llm"): 2, (requests[1].chunk_id, "embedding"): 2})
    assert read_inventory(world.connection, requests[0].chunk_id) == before
    assert all(classify_chunk(world.connection, request).state == COMPLETE for request in requests)
    world.evidence.update(first=report, resumed=resumed, successful_sibling_inventory=before,
                          provider_events=[json.loads(line) for line in events.read_text().splitlines()])


def child_command(tmp_path, requests):
    import sys

    request_path, manifest, events = [tmp_path / name for name in ("requests.json", "manifest.json", "events.jsonl")]
    request_path.write_text(json.dumps([r.model_dump(mode="json") for r in requests]))
    command = [sys.executable, str(Path(__file__).with_name("retry_ingestion_child.py")), str(request_path), str(manifest), str(events)]
    return command, request_path, manifest, events


def child_run(command, expected, *args):
    import subprocess

    result = subprocess.run([*command, *args], capture_output=True, text=True, timeout=45)
    assert result.returncode == expected, result.stderr + result.stdout
    return json.loads(result.stdout) if result.stdout.strip() else None


def test_retry_cli_drains_missing_source_and_updates_custom_manifest(world, tmp_path):
    requests = [world.request(0), world.request(1)]
    command, source, manifest, events = child_command(tmp_path, requests)
    child_run(command, 1, "--target", requests[1].chunk_id, "--fail")
    before = read_inventory(world.connection, requests[0].chunk_id)
    queue_path = str(RetryStore().for_request(requests[0]).path)
    source.unlink()
    assert not Path(requests[0].provenance.source_file).exists()
    report = child_run(command, 0, "--operation", "drain", "--queue-path", queue_path, "--chunk-id", requests[1].chunk_id)
    assert report["records_complete"] == 1 and report["chunks_selected"] == 1
    assert report["chunks_retry_attempted"] == 1 and report["chunks_newly_verified"] == 1
    assert all(c["status"] == "verified" for c in json.loads(manifest.read_text())["chunks"].values())
    assert read_inventory(world.connection, requests[0].chunk_id) == before
    inspection = child_run(command, 0, "--operation", "inspect", "--queue-path", queue_path)
    assert all(entry["retry_status"] == "resolved" for entry in inspection["entries"])
    world.evidence.update(drain=report, inspection=inspection, original_source_unavailable=True,
                          provider_events=[json.loads(line) for line in events.read_text().splitlines()])


def test_retry_cli_exhaustion_and_explicit_requeue_preserve_history(world, tmp_path):
    from collections import Counter

    requests = [world.request(0), world.request(1)]
    command, source, manifest, events = child_command(tmp_path, requests)
    cid = requests[1].chunk_id
    child_run(command, 1, "--target", cid, "--fail")
    queue_path = str(RetryStore().for_request(requests[0]).path)
    for _ in range(2):
        report = child_run(command, 1, "--operation", "drain", "--queue-path", queue_path, "--target", cid, "--fail")
    assert saved(requests[1]).status == "dead_letter"
    assert saved(requests[1]).terminal_reason == "exhausted" and len(saved(requests[1]).history) == 3
    before_events = events.read_text()
    terminal = child_run(command, 1, "--operation", "drain", "--queue-path", queue_path)
    assert terminal["chunks_dead_lettered"] == 1 and events.read_text() == before_events
    requeued = child_run(command, 0, "--operation", "requeue", "--queue-path", queue_path, "--chunk-id", cid, "--reason", "provider repaired")
    assert saved(requests[1]).status == "pending" and len(saved(requests[1]).history) == 3
    source.unlink()
    resumed = child_run(command, 0, "--operation", "drain", "--queue-path", queue_path)
    assert resumed["records_complete"] == 1
    entry = saved(requests[1])
    assert len(entry.cycles) == 2 and len(entry.history) == 4 and entry.cycle_attempts == 1
    counts = Counter((e["chunk_id"], e["kind"]) for e in map(json.loads, events.read_text().splitlines()))
    assert counts[(requests[0].chunk_id, "llm")] == counts[(requests[0].chunk_id, "embedding")] == 1
    assert counts[(cid, "llm")] == counts[(cid, "embedding")] == 4
    for request in requests:
        inventory = read_inventory(world.connection, request.chunk_id)
        assert len(inventory["nodes"]) == 5 and len(inventory["relationships"]) == 6
        assert classify_chunk(world.connection, request).state == COMPLETE
    world.evidence.update(exhausted=report, terminal=terminal, requeued=requeued, resumed=resumed,
                          final_queue=entry.model_dump(mode="json"), provider_events=[json.loads(line) for line in events.read_text().splitlines()])


@pytest.mark.parametrize("stop,exit_code", [("before_commit", 74), ("after_commit", 73), ("after_failure", 75)])
def test_retry_cli_process_death_at_safe_recovery_boundaries(world, tmp_path, stop, exit_code):
    requests = [world.request(0), world.request(1)]
    command, source, manifest, events = child_command(tmp_path, requests)
    cid = requests[1].chunk_id
    child_run(command, 1, "--target", cid, "--fail")
    before = read_inventory(world.connection, requests[0].chunk_id)
    queue_path = str(RetryStore().for_request(requests[0]).path)
    arguments = ["--operation", "drain", "--queue-path", queue_path, "--target", cid, "--stop", stop]
    if stop == "after_failure":
        arguments.append("--fail")
    child_run(command, exit_code, *arguments)
    interrupted = saved(requests[1]).model_dump(mode="json")
    assert len(interrupted["history"]) == 2
    assert interrupted["status"] == ("retry_wait" if stop == "after_failure" else "in_flight")
    assert classify_chunk(world.connection, requests[1]).state == (COMPLETE if stop == "after_commit" else ABSENT)
    provider_events = events.read_text()
    source.unlink()
    report = child_run(command, 0, "--operation", "drain", "--queue-path", queue_path)
    assert report["records_complete"] == 1 and saved(requests[1]).status == "resolved"
    if stop == "after_commit":
        assert events.read_text() == provider_events and len(saved(requests[1]).history) == 2
    else:
        assert len(saved(requests[1]).history) == 3
    assert read_inventory(world.connection, requests[0].chunk_id) == before
    assert all(c["status"] == "verified" for c in json.loads(manifest.read_text())["chunks"].values())
    world.evidence.update(stop=stop, interrupted_queue=interrupted, resumed=report,
                          provider_events=[json.loads(line) for line in events.read_text().splitlines()])


def test_requeue_cannot_bypass_unsafe_existing_graph(world, tmp_path):
    req = world.request()
    world.connection.run_write("CREATE (:Chunk {chunk_id: $id, text: $text})", {"id": req.chunk_id, "text": req.text})
    pipeline, llm, embedder = world.pipeline()
    assert asyncio.run(ingest_paragraph(world.connection, pipeline, req)).state.state == PARTIAL
    before = _snapshot(world.connection)
    command, _, _, events = child_command(tmp_path, [req])
    queue_path = str(RetryStore().for_request(req).path)
    child_run(command, 0, "--operation", "requeue", "--queue-path", queue_path, "--chunk-id", req.chunk_id)
    report = child_run(command, 1, "--operation", "drain", "--queue-path", queue_path)
    assert report["chunks_dead_lettered"] == 1 and report["records_partial"] == 1
    assert saved(req).last_error_category == "unsafe_graph_state"
    assert not events.exists() and _snapshot(world.connection) == before
    world.evidence.update(drain=report, unsafe_graph_preserved=True, provider_calls=0)


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal delivery; deterministic stop tests cover Windows control flow")
@pytest.mark.parametrize("stop_signal", [2, 15])
def test_retry_cli_clean_signal_finishes_current_chunk_before_stopping(world, tmp_path, stop_signal):
    import select
    import subprocess
    import time
    from collections import Counter

    requests = [world.request(i) for i in range(3)]
    command, source, manifest, events = child_command(tmp_path, requests)
    child_run(command, 1, "--fail-all")
    queue_path = str(RetryStore().for_request(requests[0]).path)
    process = subprocess.Popen([*command, "--operation", "drain", "--queue-path", queue_path,
                                "--pause", "--target", requests[0].chunk_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 30
        while not (tmp_path / "ready").exists():
            assert process.poll() is None, process.communicate()
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert saved(requests[0]).status == "in_flight"
        process.send_signal(stop_signal)
        # Library warnings can precede the actual handler acknowledgement.
        while True:
            assert select.select([process.stderr], [], [], 20)[0]
            if process.stderr.readline().startswith("Stop requested:"):
                break
        (tmp_path / "release").write_text("finish current pipeline")
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 130, stderr + stdout
        stopped = json.loads(stdout)
        assert stopped["stopped_early"] and stopped["chunks_processed"] == 1
        assert saved(requests[0]).status == "resolved"
        assert all(len(saved(r).history) == 1 for r in requests[1:])
        resumed = child_run(command, 0, "--operation", "drain", "--queue-path", queue_path)
        assert resumed["records_complete"] == 1
        counts = Counter((e["chunk_id"], e["kind"]) for e in map(json.loads, events.read_text().splitlines()))
        assert all(counts[(r.chunk_id, "llm")] == counts[(r.chunk_id, "embedding")] == 2 for r in requests)
        world.evidence.update(signal=stop_signal, stopped=stopped, resumed=resumed,
                              provider_events=[json.loads(line) for line in events.read_text().splitlines()])
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)


def test_pre_queue_interrupted_manifest_reconciles_commits_without_resetting_exhaustion(world, tmp_path):
    requests = [world.request(0), world.request(1)]
    manifest_path = tmp_path / "legacy-in-flight.json"
    old = ManifestStore(manifest_path=manifest_path)
    old.register(requests)
    for request in requests:
        for _ in range(3):
            old.for_request(request).mark_in_flight(request.chunk_id)
    # Reproduce a pre-queue process that committed one chunk but never wrote its
    # local checkpoint; its sibling has no graph and has already spent 3 attempts.
    pipeline, llm, embedder = world.pipeline()
    text, metadata = build_plugin_input(requests[0])
    asyncio.run(pipeline.run_async(text=text, document_metadata=metadata))
    before = read_inventory(world.connection, requests[0].chunk_id)
    outcome = loader.RunOutcome()
    with patch.object(loader, "build_simple_kg_pipeline", side_effect=AssertionError("no new provider work")):
        succeeded, failed = asyncio.run(loader.run_requests(requests, world.settings, MODELS,
            store=ManifestStore(manifest_path=manifest_path), outcome=outcome, show_progress=False))
    assert not succeeded and not failed
    assert outcome.progress.counts()["chunks_reused"] == 1
    assert outcome.progress.counts()["chunks_deferred"] == 1
    assert outcome.progress.counts()["records_partial"] == 1
    assert saved(requests[0]).status == "resolved" and saved(requests[0]).legacy_attempts == 3
    assert saved(requests[1]).status == "dead_letter" and saved(requests[1]).terminal_reason == "exhausted"
    assert saved(requests[1]).legacy_attempts == 3 and not saved(requests[1]).history
    assert llm.calls == embedder.calls == 1
    assert read_inventory(world.connection, requests[0].chunk_id) == before
    world.evidence.update(outcome=outcome.to_dict(), queue=[saved(r).model_dump(mode="json") for r in requests],
                          committed_inventory=before, provider_calls=llm.calls, legacy_budget_preserved=True)
