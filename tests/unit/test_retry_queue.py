"""Durable queue contract tests; no Neo4j or provider service calls."""

import json
import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from graphrag_pipeline.extraction.schemas import ExtractionRequest, Provenance
from graphrag_pipeline.ingestion.retry_policy import FailureDecision, RetryPolicy
from graphrag_pipeline.ingestion.retry_queue import QueueError, RetryQueue, default_queue_path


SHA = "a" * 64
TRANSIENT = FailureDecision("timeout", True)
TERMINAL = FailureDecision("authentication", False)


def request(idx=0, record="r1", sha=SHA, split="dev"):
    return ExtractionRequest(
        chunk_id=f"musique:{record}:paragraph:{idx}", record_id=record,
        document_id="doc", paragraph_idx=idx, text=f"  Paragraph {idx}.\n", title="café",
        provenance=Provenance(source_dataset="musique", source_split=split,
                              source_file="missing.jsonl", source_sha256=sha, source_line=1),
    )


@pytest.fixture
def world(tmp_path):
    now = [100.0]
    path = tmp_path / "retry.json"
    queue = RetryQueue.open(path, SHA, "dev", clock=lambda: now[0])
    requests = [request(), request(1)]
    queue.register(requests)
    return queue, requests, now


def reopen(queue, now):
    return RetryQueue.open(queue.path, SHA, "dev", clock=lambda: now[0])


def test_registration_persists_full_siblings_without_original_dataset(world):
    queue, requests, now = world
    loaded = reopen(queue, now).snapshot()
    assert loaded.records == {"r1": sorted(r.chunk_id for r in requests)}
    assert [e.request for e in loaded.chunks.values()] == requests
    assert all(e.status == "pending" and not e.history for e in loaded.chunks.values())
    assert not queue.path.with_name("missing.jsonl").exists()
    snapshot = queue.snapshot()
    snapshot.chunks.clear()
    assert len(queue.snapshot().chunks) == 2


def test_backoff_and_exhaustion_survive_repeated_restarts(world):
    queue, requests, now = world
    cid = requests[0].chunk_id
    for number, delay in [(1, 5), (2, 30), (3, None)]:
        queue.begin_attempt(cid, {"provider": "ollama", "llm_model": f"model-{number}"})
        assert cid not in queue.due_ids()
        queue.fail_attempt(cid, TRANSIENT, "extraction")
        queue = reopen(queue, now)
        entry = queue.snapshot().chunks[cid]
        assert len(entry.history) == number
        assert entry.last_error_category == "timeout"
        if delay is not None:
            assert entry.status == "retry_wait" and entry.next_attempt_at == now[0] + delay
            now[0] += delay - 0.001
            assert cid not in queue.due_ids()
            now[0] += 0.001
            assert cid in queue.due_ids()
        else:
            assert entry.status == "dead_letter" and entry.terminal_reason == "exhausted"
            now[0] += 10000
            with pytest.raises(QueueError, match="not eligible"):
                queue.begin_attempt(cid, {})


def test_terminal_requeue_preserves_history_and_uses_new_bounded_cycle(world):
    queue, requests, now = world
    cid = requests[0].chunk_id
    queue.begin_attempt(cid, {})
    queue.fail_attempt(cid, TERMINAL, "embedding")
    entry = reopen(queue, now).snapshot().chunks[cid]
    assert entry.terminal_reason == "non_retryable" and entry.history[0].stage == "embedding"
    with pytest.raises(QueueError):
        queue.requeue(cid, "  ")
    queue.requeue(cid, "credentials fixed", RetryPolicy(max_attempts=1, delays_seconds=()))
    queue.begin_attempt(cid, {"embedding_model": "replacement"})
    queue.fail_attempt(cid, TRANSIENT, "embedding")
    entry = reopen(queue, now).snapshot().chunks[cid]
    assert [(a.number, a.cycle, a.cycle_attempt) for a in entry.history] == [(1, 1, 1), (2, 2, 1)]
    assert entry.terminal_reason == "exhausted"
    assert entry.cycles[1].reason == "credentials fixed"


def test_registered_identity_and_policy_do_not_reset(world):
    queue, requests, now = world
    cid = requests[0].chunk_id
    queue.begin_attempt(cid, {})
    queue.fail_attempt(cid, TRANSIENT, "write")
    before = queue.path.read_bytes()
    queue.register(requests, RetryPolicy(max_attempts=1, delays_seconds=()))
    assert queue.path.read_bytes() == before
    for field, value in [("text", "changed"), ("title", "changed"), ("document_id", "changed")]:
        with pytest.raises(QueueError, match="conflicting"):
            queue.register([requests[0].model_copy(update={field: value}), requests[1]])
        assert queue.path.read_bytes() == before
    with pytest.raises(QueueError, match="inventory changed"):
        queue.register(requests[:1])
    assert queue.path.read_bytes() == before


@pytest.mark.parametrize("sha,split", [("b" * 64, "dev"), (SHA, "train")])
def test_open_rejects_source_mismatch_without_touching_file(world, sha, split):
    queue, _, _ = world
    before = queue.path.read_bytes()
    with pytest.raises(QueueError, match="source checksum or split mismatch"):
        RetryQueue.open(queue.path, sha, split)
    assert queue.path.read_bytes() == before


def test_register_rejects_wrong_source_without_touching_file(world):
    queue, _, _ = world
    before = queue.path.read_bytes()
    with pytest.raises(QueueError, match="different source"):
        queue.register([request(record="other", sha="b" * 64)])
    assert queue.path.read_bytes() == before


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(version=2),
    lambda d: d.update(extra="unexpected"),
    lambda d: d["chunks"]["musique:r1:paragraph:0"].update(request_fingerprint="tampered"),
    lambda d: d["chunks"]["musique:r1:paragraph:0"]["request"].update(text="changed"),
    lambda d: d["records"]["r1"].append("missing"),
    lambda d: d["records"]["r1"].append("musique:r1:paragraph:0"),
    lambda d: d["records"].clear(),
    lambda d: d["chunks"]["musique:r1:paragraph:0"].update(status="in_flight"),
    lambda d: d["chunks"]["musique:r1:paragraph:0"].update(next_attempt_at=float("nan")),
    lambda d: d["chunks"]["musique:r1:paragraph:0"].update(status="dead_letter", next_attempt_at=None),
])
def test_corruption_is_rejected_and_preserved(world, mutation):
    queue, _, now = world
    data = json.loads(queue.path.read_text())
    mutation(data)
    queue.path.write_text(json.dumps(data))
    before = queue.path.read_bytes()
    with pytest.raises(QueueError):
        reopen(queue, now)
    assert queue.path.read_bytes() == before


@pytest.mark.parametrize("content", ['{', '[]', '{"version":1,"version":1}'])
def test_unreadable_json_is_preserved(world, content):
    queue, _, now = world
    queue.path.write_text(content)
    with pytest.raises(QueueError):
        reopen(queue, now)
    assert queue.path.read_text() == content


def test_atomic_replace_failure_keeps_old_file_and_poisoned_instance(world):
    queue, requests, now = world
    before = queue.path.read_bytes()
    with patch("graphrag_pipeline.ingestion.retry_queue.os.replace", side_effect=OSError("disk failure")):
        with pytest.raises(QueueError, match="cannot persist"):
            queue.begin_attempt(requests[0].chunk_id, {})
    assert queue.path.read_bytes() == before
    assert list(queue.path.parent.glob("*.tmp")) == []
    assert queue.snapshot().chunks[requests[0].chunk_id].history == []
    with pytest.raises(QueueError, match="reopen"):
        queue.begin_attempt(requests[0].chunk_id, {})
    reopen(queue, now).begin_attempt(requests[0].chunk_id, {})


def test_fsync_file_and_directory_and_failure_after_replace(world):
    queue, requests, now = world
    cid = requests[0].chunk_id
    actual = os.fsync
    calls = []

    def fsync(fd):
        calls.append(fd)
        if len(calls) == 2:
            raise OSError("directory sync failure")
        return actual(fd)

    with patch("graphrag_pipeline.ingestion.retry_queue.os.fsync", side_effect=fsync):
        with pytest.raises(QueueError):
            queue.begin_attempt(cid, {})
    assert len(calls) == 2
    assert reopen(queue, now).snapshot().chunks[cid].status == "in_flight"
    with pytest.raises(QueueError, match="reopen"):
        queue.resolve_verified(cid)


def test_interruption_requires_reconciliation_and_keeps_attempt_budget(world):
    queue, requests, now = world
    cid = requests[0].chunk_id
    queue.begin_attempt(cid, {})
    queue = reopen(queue, now)
    assert queue.snapshot().chunks[cid].status == "in_flight"
    with pytest.raises(QueueError):
        queue.begin_attempt(cid, {})
    queue.reconcile_absent(cid)
    entry = reopen(queue, now).snapshot().chunks[cid]
    assert entry.status == "retry_wait" and entry.next_attempt_at == 105
    assert len(entry.history) == 1 and entry.history[0].outcome == "interrupted"


def test_verified_commit_resolves_without_an_additional_attempt(world):
    queue, requests, now = world
    cid = requests[0].chunk_id
    queue.begin_attempt(cid, {})
    reopen(queue, now).resolve_verified(cid)
    entry = reopen(queue, now).snapshot().chunks[cid]
    assert entry.status == "resolved" and len(entry.history) == 1
    assert entry.history[0].outcome == "verified"
    # Already committed siblings can resolve directly after Neo4j verification.
    queue = reopen(queue, now)
    queue.resolve_verified(requests[1].chunk_id)
    assert not queue.snapshot().chunks[requests[1].chunk_id].history
    assert not queue.due_ids()


def test_configuration_rejects_secret_keys_before_persistence(world):
    queue, requests, _ = world
    before = queue.path.read_bytes()
    with pytest.raises(ValidationError, match="secret-bearing"):
        queue.begin_attempt(requests[0].chunk_id, {"api_key": "do-not-persist"})
    assert queue.path.read_bytes() == before


def test_path_is_source_scoped_and_split_collision_safe():
    assert default_queue_path(SHA, "a/b") != default_queue_path(SHA, "a_b")
    assert default_queue_path(SHA, "dev") != default_queue_path("a" * 63 + "b", "dev")
    with pytest.raises(ValidationError):
        default_queue_path("", "dev")


def test_due_selection_is_stable_and_scoped(world):
    queue, requests, _ = world
    queue.register([request(record="r2")])
    assert queue.due_ids(record_id="r1") == sorted(r.chunk_id for r in requests)
    assert len(queue.due_ids()) == 3


@pytest.mark.parametrize("mutation", [
    lambda e: e.update(last_error_category="different"),
    lambda e: e.update(status="dead_letter", next_attempt_at=None, terminal_reason="exhausted"),
    lambda e: e["history"][0].update(number=9),
    lambda e: e["history"][0].update(cycle=9),
    lambda e: e["history"][0].update(cycle_attempt=2),
    lambda e: e["history"][0].update(finished_at=None),
    lambda e: e["history"][0].update(configuration={"password": "secret"}),
    lambda e: e["history"][0].update(retryable=False),
    lambda e: e.update(next_attempt_at=101),
    lambda e: e.update(status="pending"),
    lambda e: e["cycles"][0].update(policy={"max_attempts": 1, "delays_seconds": []}),
])
def test_tampered_history_cannot_bypass_attempt_budget_or_backoff(world, mutation):
    queue, requests, now = world
    cid = requests[0].chunk_id
    queue.begin_attempt(cid, {})
    queue.fail_attempt(cid, TRANSIENT, "extraction")
    data = json.loads(queue.path.read_text())
    mutation(data["chunks"][cid])
    queue.path.write_text(json.dumps(data))
    before = queue.path.read_bytes()
    with pytest.raises(QueueError):
        reopen(queue, now)
    assert queue.path.read_bytes() == before


def test_file_sync_failure_preserves_previous_state(world):
    queue, requests, now = world
    before = queue.path.read_bytes()
    with patch("graphrag_pipeline.ingestion.retry_queue.os.fsync", side_effect=OSError("disk full")):
        with pytest.raises(QueueError):
            queue.begin_attempt(requests[0].chunk_id, {})
    assert queue.path.read_bytes() == before
    assert not list(queue.path.parent.glob("*.tmp"))
    assert reopen(queue, now).due_ids() == sorted(r.chunk_id for r in requests)


def test_registration_is_all_or_nothing(world):
    queue, requests, _ = world
    before = queue.path.read_bytes()
    with pytest.raises(QueueError):
        queue.register([request(record="r2"), requests[0].model_copy(update={"text": "conflict"})])
    assert queue.path.read_bytes() == before
    assert "r2" not in queue.snapshot().records


def test_new_process_reads_in_flight_journal_without_silent_demotion(world):
    import subprocess
    import sys

    queue, requests, _ = world
    cid = requests[0].chunk_id
    queue.begin_attempt(cid, {"provider": "ollama"})
    child = subprocess.run([
        sys.executable, "-c",
        "from pathlib import Path; import sys; "
        "from graphrag_pipeline.ingestion.retry_queue import RetryQueue; "
        "q=RetryQueue.open(Path(sys.argv[1]), sys.argv[2], 'dev'); "
        "e=q.snapshot().chunks[sys.argv[3]]; "
        "assert e.status=='in_flight' and len(e.history)==1; "
        "assert e.request.text=='  Paragraph 0.\\n'; "
        "print('durable in-flight request verified')",
        str(queue.path), SHA, cid,
    ], capture_output=True, text=True, check=True)
    assert child.stdout.strip() == "durable in-flight request verified"


def test_single_attempt_cycle_exhausts_after_interruption(tmp_path):
    queue = RetryQueue.open(tmp_path / "q.json", SHA, "dev", clock=lambda: 100)
    req = request()
    queue.register([req], RetryPolicy(max_attempts=1, delays_seconds=()))
    queue.begin_attempt(req.chunk_id, {})
    queue.reconcile_absent(req.chunk_id)
    assert queue.snapshot().chunks[req.chunk_id].terminal_reason == "exhausted"


def test_failed_and_resolved_entries_cannot_be_replayed_directly(world):
    queue, requests, _ = world
    cid = requests[0].chunk_id
    with pytest.raises(QueueError):
        queue.fail_attempt(cid, TRANSIENT, "extraction")
    with pytest.raises(QueueError):
        queue.reconcile_absent(cid)
    with pytest.raises(QueueError):
        queue.requeue(cid, "not a dead letter")
    queue.resolve_verified(cid)
    with pytest.raises(QueueError):
        queue.begin_attempt(cid, {})


@pytest.mark.parametrize("now", [float("nan"), float("inf"), -1])
def test_invalid_clock_cannot_schedule_work(world, now):
    queue, _, clock = world
    clock[0] = now
    before = queue.path.read_bytes()
    with pytest.raises(QueueError, match="clock"):
        queue.due_ids()
    assert queue.path.read_bytes() == before


def test_boolean_version_is_rejected(world):
    queue, _, now = world
    data = json.loads(queue.path.read_text())
    data["version"] = True
    queue.path.write_text(json.dumps(data))
    with pytest.raises(QueueError):
        reopen(queue, now)


def test_old_reconciliation_error_cannot_override_newer_attempt_backoff(world):
    queue, requests, now = world
    cid = requests[0].chunk_id
    queue.record_observation(cid, FailureDecision("connectivity", True), "reconcile")
    now[0] += 5
    queue.begin_attempt(cid, {})
    queue.fail_attempt(cid, FailureDecision("malformed_model_output", True), "extraction")
    persisted = json.loads(queue.path.read_text())
    assert persisted["chunks"][cid]["next_attempt_at"] == 110
    persisted["chunks"][cid]["last_error_category"] = "connectivity"
    persisted["chunks"][cid]["next_attempt_at"] = 105
    queue.path.write_text(json.dumps(persisted))
    with pytest.raises(QueueError, match="last error category"):
        reopen(queue, now)
