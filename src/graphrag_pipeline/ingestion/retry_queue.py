"""Source-scoped atomic retry journal for sequential paragraph ingestion.

Queue state is operational evidence, never proof of Neo4j completion. Callers
must reconcile interrupted and resolved entries against Neo4j before replay or
reuse. Progress manifests cannot reset this independent retry journal.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr, model_validator

from graphrag_pipeline.extraction.idempotency import fingerprint
from graphrag_pipeline.extraction.schemas import ExtractionRequest
from graphrag_pipeline.ingestion.retry_policy import FailureDecision, RetryPolicy, Stage

DEFAULT_RETRY_DIR = Path("runs/ingestion-retries")
Number = StrictInt | StrictFloat
Status = Literal["pending", "in_flight", "retry_wait", "dead_letter", "resolved"]
# Values accepted from configured settings, never credentials, arbitrary kwargs,
# provider response bodies or user-supplied diagnostic strings.
CONFIG_KEYS = {"provider", "llm_model", "embedding_model", "embedding_dimensions",
               "llm_max_tokens", "database", "ingestion_contract_version"}


class QueueError(RuntimeError):
    """Invalid queue, identity mismatch or persistence failure; stop processing."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Attempt(_Model):
    number: StrictInt = Field(ge=1)
    cycle: StrictInt = Field(ge=1)
    cycle_attempt: StrictInt = Field(ge=1)
    started_at: Number = Field(ge=0)
    finished_at: Number | None = Field(default=None, ge=0)
    outcome: Literal["in_flight", "failed", "interrupted", "verified"] = "in_flight"
    stage: Stage | None = None
    category: StrictStr | None = None
    retryable: StrictBool | None = None
    configuration: dict[str, StrictStr | StrictInt] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_attempt(self):
        if set(self.configuration) - CONFIG_KEYS:
            raise ValueError("attempt configuration contains unsupported or secret-bearing keys")
        if (self.outcome == "in_flight") != (self.finished_at is None):
            raise ValueError("attempt outcome and finish timestamp disagree")
        if self.finished_at is not None and self.finished_at < self.started_at:
            raise ValueError("attempt finishes before it starts")
        if self.outcome == "failed" and (not self.category or self.stage is None or self.retryable is None):
            raise ValueError("failed attempt requires a classified error")
        return self


class RetryCycle(_Model):
    policy: RetryPolicy
    started_at: Number = Field(ge=0)
    reason: StrictStr = Field(min_length=1)


class FailureObservation(_Model):
    after_attempt: StrictInt = Field(ge=0)
    at: Number = Field(ge=0)
    stage: Stage
    category: StrictStr = Field(min_length=1)
    retryable: StrictBool


class QueueEntry(_Model):
    request: ExtractionRequest
    request_fingerprint: StrictStr
    status: Status = "pending"
    cycles: list[RetryCycle] = Field(min_length=1)
    history: list[Attempt] = Field(default_factory=list)
    observations: list[FailureObservation] = Field(default_factory=list)
    legacy_attempts: StrictInt = Field(default=0, ge=0)
    last_error_category: StrictStr | None = None
    terminal_reason: StrictStr | None = None
    next_attempt_at: Number | None = Field(default=None, ge=0)

    @property
    def cycle_attempts(self) -> int:
        return sum(a.cycle == len(self.cycles) for a in self.history) + (self.legacy_attempts if len(self.cycles) == 1 else 0)

    @model_validator(mode="after")
    def valid_entry(self):
        if self.request_fingerprint != fingerprint(self.request):
            raise ValueError("request fingerprint mismatch")
        counts: dict[int, int] = {1: self.legacy_attempts}
        previous_cycle = 0
        for number, attempt in enumerate(self.history, self.legacy_attempts + 1):
            if attempt.number != number or not previous_cycle <= attempt.cycle <= len(self.cycles):
                raise ValueError("invalid attempt sequence")
            counts[attempt.cycle] = counts.get(attempt.cycle, 0) + 1
            if attempt.cycle_attempt != counts[attempt.cycle]:
                raise ValueError("invalid cycle attempt sequence")
            if counts[attempt.cycle] > self.cycles[attempt.cycle - 1].policy.max_attempts:
                raise ValueError("attempt budget exceeded")
            previous_cycle = attempt.cycle
            if attempt.outcome == "in_flight" and number != len(self.history) + self.legacy_attempts:
                raise ValueError("unfinished historical attempt")
        in_flight = bool(self.history and self.history[-1].outcome == "in_flight")
        if (self.status == "in_flight") != in_flight:
            raise ValueError("in-flight status and history disagree")
        if (self.status in {"pending", "retry_wait"}) != (self.next_attempt_at is not None):
            raise ValueError("retry eligibility timestamp and status disagree")
        if (self.status == "dead_letter") != (self.terminal_reason is not None):
            raise ValueError("dead letter requires a terminal reason")
        if self.status in {"in_flight", "retry_wait"} and self.history and self.history[-1].cycle != len(self.cycles):
            raise ValueError("active attempt belongs to an older retry cycle")
        events = [((a.number, 0), a.finished_at, a.retryable if a.outcome == "failed" else True, a.category)
                  for a in self.history if a.outcome in {"failed", "interrupted"}]
        previous_observation = 0
        for index, observation in enumerate(self.observations, 1):
            if not previous_observation <= observation.after_attempt <= len(self.history) + self.legacy_attempts:
                raise ValueError("observation disagrees with attempt history")
            previous_observation = observation.after_attempt
            events.append(((observation.after_attempt, index), observation.at, observation.retryable, observation.category))
        latest = max(events, key=lambda event: event[0]) if events else None
        if latest is not None and self.last_error_category != latest[3]:
            raise ValueError("last error category disagrees with attempt history")
        if self.terminal_reason == "exhausted" and self.cycle_attempts < self.cycles[-1].policy.max_attempts:
            raise ValueError("dead letter claims exhaustion before its budget is used")
        if self.status == "pending" and self.cycle_attempts:
            raise ValueError("pending cycle already has attempts")
        if self.status == "retry_wait":
            if self.cycle_attempts >= self.cycles[-1].policy.max_attempts:
                raise ValueError("exhausted entry cannot wait for retry")
            if latest is None:
                raise ValueError("retry wait requires a failed or interrupted attempt or observation")
            _, finished, retryable, _ = latest
            if not retryable:
                raise ValueError("terminal failure cannot wait for retry")
            delay = self.cycles[-1].policy.delay_after(max(1, self.cycle_attempts)) if self.cycles[-1].policy.max_attempts > 1 else 0
            if self.next_attempt_at < finished + delay:
                raise ValueError("retry timestamp bypasses backoff")
        return self


class QueueData(_Model):
    version: StrictInt = 1
    source_sha256: StrictStr = Field(min_length=1)
    source_split: StrictStr = Field(min_length=1)
    records: dict[str, list[StrictStr]] = Field(default_factory=dict)
    chunks: dict[str, QueueEntry] = Field(default_factory=dict)
    manifest_paths: list[StrictStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_source(self):
        if self.version != 1:
            raise ValueError("unsupported retry queue version")
        if len(self.manifest_paths) != len(set(self.manifest_paths)) or any(not Path(p).is_absolute() for p in self.manifest_paths):
            raise ValueError("manifest references must be unique absolute paths")
        expected = []
        for record_id, ids in self.records.items():
            if not ids or len(ids) != len(set(ids)):
                raise ValueError("empty or duplicate expected chunk inventory")
            for chunk_id in ids:
                if chunk_id not in self.chunks or self.chunks[chunk_id].request.record_id != record_id:
                    raise ValueError("record inventory disagrees with stored requests")
            expected.extend(ids)
        if len(expected) != len(set(expected)) or set(expected) != set(self.chunks):
            raise ValueError("orphan or repeated queue entry")
        for chunk_id, entry in self.chunks.items():
            provenance = entry.request.provenance
            if chunk_id != entry.request.chunk_id:
                raise ValueError("queue key differs from request chunk_id")
            if (provenance.source_sha256, provenance.source_split) != (self.source_sha256, self.source_split):
                raise ValueError("request belongs to a different source")
        return self


def default_queue_path(source_sha256: str, source_split: str, directory: Path = DEFAULT_RETRY_DIR) -> Path:
    """Use full digests of both source fields to avoid truncated/sanitized collisions."""
    QueueData(source_sha256=source_sha256, source_split=source_split)
    split_digest = hashlib.sha256(source_split.encode("utf-8")).hexdigest()
    # Historical synthetic fixtures use opaque provenance tokens. Preserve
    # their exact identity internally while making path construction safe.
    import re
    source_key = source_sha256 if re.fullmatch(r"[0-9a-f]{64}", source_sha256) else "opaque-" + hashlib.sha256(source_sha256.encode()).hexdigest()
    return directory / f"{source_key}-{split_digest}.json"


def _write_atomic(path: Path, data: QueueData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data.model_dump_json(indent=2))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


class RetryQueue:
    """Copy-on-write queue. All mutation methods persist before returning.

    A persistence failure poisons this instance: reopen from disk before further
    work because a failed directory fsync may follow a successful replacement.
    Opening never demotes in-flight state; reconciliation is the caller's job.
    """

    def __init__(self, path: Path, data: QueueData, clock: Callable[[], float] = time.time):
        self.path, self._data, self._clock = path, data, clock
        self._failed = False

    @classmethod
    def open(cls, path: Path, source_sha256: str, source_split: str, *, clock=time.time):
        try:
            expected = QueueData(source_sha256=source_sha256, source_split=source_split)
            data = QueueData.model_validate(json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)) if path.exists() else expected
            if (data.source_sha256, data.source_split) != (source_sha256, source_split):
                raise ValueError("queue source checksum or split mismatch")
        except (OSError, ValueError) as error:
            raise QueueError(f"cannot open retry queue {path}: {error}") from error
        return cls(path, data, clock)

    @classmethod
    def load(cls, path: Path, *, clock=time.time):
        """Load an existing self-describing queue; never create one implicitly."""
        try:
            data = QueueData.model_validate(json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object))
        except (OSError, ValueError) as error:
            raise QueueError(f"cannot load retry queue {path}: {error}") from error
        return cls(path, data, clock)

    def remember_manifest(self, path: Path):
        reference = str(path.resolve())
        if reference not in self._data.manifest_paths:
            data = self.snapshot()
            data.manifest_paths.append(reference)
            data.manifest_paths.sort()
            self._commit(data)

    def snapshot(self) -> QueueData:
        return self._data.model_copy(deep=True)

    def _now(self) -> float:
        now = self._clock()
        if not math.isfinite(now) or now < 0:
            raise QueueError("queue clock must return a finite nonnegative timestamp")
        return float(now)

    def _commit(self, data: QueueData):
        if self._failed:
            raise QueueError("queue persistence previously failed; reopen before continuing")
        try:
            checked = QueueData.model_validate(data.model_dump(mode="python"))
            _write_atomic(self.path, checked)
        except OSError as error:
            self._failed = True
            raise QueueError(f"cannot persist retry queue {self.path}: {error}") from error
        self._data = checked

    def register(self, requests: list[ExtractionRequest], policy: RetryPolicy | None = None):
        """Register complete expected inventories for the selected records.

        Registration is not an attempt. Known requests and budgets never reset.
        """
        data, policy, now = self.snapshot(), policy or RetryPolicy(), self._now()
        groups: dict[str, list[str]] = {}
        for request in requests:
            request = ExtractionRequest.model_validate(request.model_dump())
            groups.setdefault(request.record_id, []).append(request.chunk_id)
            existing = data.chunks.get(request.chunk_id)
            if existing is not None and existing.request_fingerprint != fingerprint(request):
                raise QueueError(f"conflicting request identity for {request.chunk_id}")
            if existing is None:
                data.chunks[request.chunk_id] = QueueEntry(
                    request=request, request_fingerprint=fingerprint(request), next_attempt_at=now,
                    cycles=[RetryCycle(policy=policy, started_at=now, reason="initial")],
                )
        for record_id, ids in groups.items():
            ids = sorted(ids)
            if record_id in data.records and not set(data.records[record_id]).issubset(ids):
                raise QueueError(f"expected chunk inventory changed for {record_id}")
            data.records[record_id] = ids
        try:
            self._commit(data)
        except ValueError as error:
            raise QueueError(f"invalid queue registration: {error}") from error

    def due_ids(self, *, record_id: str | None = None) -> list[str]:
        now = self._now()
        return sorted(cid for cid, e in self._data.chunks.items()
                      if e.status in {"pending", "retry_wait"} and e.next_attempt_at <= now
                      and (record_id is None or e.request.record_id == record_id))

    def begin_attempt(self, chunk_id: str, configuration: dict[str, str | int]):
        data, now = self.snapshot(), self._now()
        entry = data.chunks[chunk_id]
        if chunk_id not in self.due_ids() or entry.cycle_attempts >= entry.cycles[-1].policy.max_attempts:
            raise QueueError("entry is not eligible for an attempt")
        entry.history.append(Attempt(number=len(entry.history) + entry.legacy_attempts + 1, cycle=len(entry.cycles),
                                     cycle_attempt=entry.cycle_attempts + 1, started_at=now,
                                     configuration=configuration))
        entry.status, entry.next_attempt_at = "in_flight", None
        self._commit(data)

    def fail_attempt(self, chunk_id: str, decision: FailureDecision, stage: Stage):
        data, now = self.snapshot(), self._now()
        entry = data.chunks[chunk_id]
        if entry.status != "in_flight":
            raise QueueError("only an in-flight attempt may fail")
        attempt = entry.history[-1]
        attempt.outcome, attempt.finished_at = "failed", max(now, attempt.started_at)
        attempt.category, attempt.retryable, attempt.stage = decision.category, decision.retryable, stage
        entry.last_error_category = decision.category
        self._schedule(entry, max(now, attempt.started_at), decision.retryable)
        self._commit(data)

    @staticmethod
    def _schedule(entry, now, retryable):
        if not retryable or entry.cycle_attempts >= entry.cycles[-1].policy.max_attempts:
            entry.status, entry.next_attempt_at = "dead_letter", None
            entry.terminal_reason = "exhausted" if retryable else "non_retryable"
        else:
            entry.status = "retry_wait"
            entry.next_attempt_at = now + entry.cycles[-1].policy.delay_after(entry.cycle_attempts)

    def reconcile_absent(self, chunk_id: str):
        """Call only after read-only Neo4j verification proves interrupted work absent."""
        data, now = self.snapshot(), self._now()
        entry = data.chunks[chunk_id]
        if entry.status != "in_flight":
            raise QueueError("absent recovery requires an interrupted in-flight entry")
        attempt = entry.history[-1]
        attempt.outcome, attempt.finished_at = "interrupted", max(now, attempt.started_at)
        attempt.category = entry.last_error_category = "interrupted"
        self._schedule(entry, max(now, attempt.started_at), True)
        self._commit(data)

    def resolve_verified(self, chunk_id: str):
        """Call only after the authoritative Neo4j completion inventory verifies."""
        data, now = self.snapshot(), self._now()
        entry = data.chunks[chunk_id]
        if entry.status == "in_flight":
            entry.history[-1].outcome = "verified"
            entry.history[-1].finished_at = max(now, entry.history[-1].started_at)
        entry.status, entry.next_attempt_at, entry.terminal_reason = "resolved", None, None
        self._commit(data)

    def requeue(self, chunk_id: str, reason: str, policy: RetryPolicy | None = None):
        self.requeue_many([chunk_id], reason, policy)

    def requeue_many(self, chunk_ids: list[str], reason: str, policy: RetryPolicy | None = None):
        """Validate and persist all targeted cycle changes in one atomic update."""
        if not chunk_ids or not reason.strip():
            raise QueueError("requeue requires dead letters and a nonempty operator reason")
        data, now = self.snapshot(), self._now()
        for chunk_id in sorted(set(chunk_ids)):
            if chunk_id not in data.chunks or data.chunks[chunk_id].status != "dead_letter":
                raise QueueError(f"requeue requires a dead letter: {chunk_id}")
            entry = data.chunks[chunk_id]
            entry.cycles.append(RetryCycle(policy=policy or RetryPolicy(), started_at=now, reason=reason))
            entry.status, entry.next_attempt_at, entry.terminal_reason = "pending", now, None
        self._commit(data)

    def ensure_request(self, request: ExtractionRequest):
        """Singleton/fixture calls may reuse an existing full record inventory."""
        existing = self._data.chunks.get(request.chunk_id)
        if existing is not None:
            if existing.request_fingerprint != fingerprint(request):
                raise QueueError(f"conflicting request identity for {request.chunk_id}")
            return
        siblings = [self._data.chunks[cid].request for cid in self._data.records.get(request.record_id, [])]
        self.register([*siblings, request], RetryPolicy.from_env())

    def record_observation(self, chunk_id: str, decision: FailureDecision, stage: Stage):
        """Persist a failure outside provider work without consuming an attempt."""
        data, now = self.snapshot(), self._now()
        entry = data.chunks[chunk_id]
        entry.observations.append(FailureObservation(after_attempt=len(entry.history) + entry.legacy_attempts, at=now, stage=stage,
                                                     category=decision.category, retryable=decision.retryable))
        was_terminal = entry.status == "dead_letter"
        entry.last_error_category = decision.category
        if was_terminal:
            # Recording a read failure cannot reauthorize terminal provider work.
            self._commit(data)
            return
        if entry.status == "in_flight":
            attempt = entry.history[-1]
            attempt.outcome, attempt.finished_at = "interrupted", max(now, attempt.started_at)
            attempt.category = decision.category
        if not decision.retryable or entry.cycle_attempts >= entry.cycles[-1].policy.max_attempts:
            entry.status, entry.next_attempt_at = "dead_letter", None
            entry.terminal_reason = "exhausted" if decision.retryable else "non_retryable"
        else:
            entry.status, entry.terminal_reason = "retry_wait", None
            delay = entry.cycles[-1].policy.delay_after(max(1, entry.cycle_attempts)) if entry.cycles[-1].policy.max_attempts > 1 else 0
            entry.next_attempt_at = now + delay
        self._commit(data)

    def import_legacy_failure(self, chunk_id: str, attempts: int):
        """Old diagnostics cannot prove a safe typed cause; retain for review.

        Preserve lifetime attempts without fabricating individual provider runs.
        Unknown legacy failures are dead letters until explicitly requeued.
        """
        entry = self._data.chunks[chunk_id]
        if entry.history or entry.observations or entry.legacy_attempts or entry.status != "pending":
            return
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 0:
            raise QueueError("invalid legacy attempt count")
        data = self.snapshot()
        entry = data.chunks[chunk_id]
        entry.legacy_attempts = attempts
        entry.status, entry.next_attempt_at = "dead_letter", None
        entry.terminal_reason = "non_retryable"
        entry.last_error_category = "legacy_unclassified"
        entry.observations.append(FailureObservation(after_attempt=attempts, at=self._now(), stage="extraction",
                                                     category="legacy_unclassified", retryable=False))
        self._commit(data)


    def import_legacy_interruption(self, chunk_id: str, attempts: int):
        """Preserve pre-queue in-flight budgets without inventing provider history.

        Neo4j reconciliation still happens before any replay, even when an old
        count already exhausts the new policy. Matching commits can resolve it.
        """
        entry = self._data.chunks[chunk_id]
        if entry.history or entry.observations or entry.legacy_attempts or entry.status != "pending":
            return
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
            raise QueueError("invalid legacy interrupted attempt count")
        data, now = self.snapshot(), self._now()
        entry = data.chunks[chunk_id]
        entry.legacy_attempts = attempts
        entry.last_error_category = "legacy_interrupted"
        entry.observations.append(FailureObservation(after_attempt=attempts, at=now, stage="reconcile",
                                                     category="legacy_interrupted", retryable=True))
        if attempts >= entry.cycles[-1].policy.max_attempts:
            entry.status, entry.next_attempt_at, entry.terminal_reason = "dead_letter", None, "exhausted"
        else:
            entry.status = "retry_wait"
            entry.next_attempt_at = now + entry.cycles[-1].policy.delay_after(attempts)
        self._commit(data)


class RetryStore:
    """Resolve one queue per source; independent from manifest path/freshness."""

    def __init__(self, directory: Path | None = None, *, clock=time.time):
        self.directory = directory if directory is not None else Path(os.environ.get("INGESTION_RETRY_STATE_DIR", str(DEFAULT_RETRY_DIR)))
        self._queues: dict[tuple[str, str], RetryQueue] = {}
        self._clock = clock

    def for_request(self, request: ExtractionRequest) -> RetryQueue:
        key = (request.provenance.source_sha256, request.provenance.source_split)
        if key not in self._queues:
            self._queues[key] = RetryQueue.open(default_queue_path(*key, self.directory), *key, clock=self._clock)
        return self._queues[key]

    def register(self, requests: list[ExtractionRequest], manifests=None):
        groups: dict[tuple[str, str], list[ExtractionRequest]] = {}
        for request in requests:
            key = (request.provenance.source_sha256, request.provenance.source_split)
            groups.setdefault(key, []).append(request)
        for group in groups.values():
            queue = self.for_request(group[0])
            queue.register(group, RetryPolicy.from_env())
            if manifests is not None:
                for request in group:
                    old = manifests.existing_failure_snapshot(request)
                    if old is not None and old["status"] == "failed":
                        queue.import_legacy_failure(request.chunk_id, old.get("attempts", 0))
                    elif old is not None and (old["status"] == "in_flight" or old.get("interrupted")):
                        queue.import_legacy_interruption(request.chunk_id, old.get("attempts", 0))
        if manifests is not None:
            # Failure history is durable in the queue before --fresh-manifest
            # can archive or replace progress. A crash cannot reset its budget.
            manifests.register(requests)
            for group in groups.values():
                self.for_request(group[0]).remember_manifest(manifests.for_request(group[0]).path)

    def record_is_resolved(self, request: ExtractionRequest) -> bool:
        data = self.for_request(request).snapshot()
        return all(data.chunks[cid].status == "resolved" for cid in data.records[request.record_id])
