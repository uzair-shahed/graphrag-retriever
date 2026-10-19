"""Per-invocation counts derived from exact source/record/chunk identities."""

from collections import Counter
from dataclasses import dataclass, field


def source_key(request):
    return request.provenance.source_sha256, request.provenance.source_split


def chunk_key(request):
    return (*source_key(request), request.chunk_id)


@dataclass
class RunProgress:
    records: dict = field(default_factory=dict)
    outcomes: dict = field(default_factory=dict)
    retry_attempts: int = 0
    queue_paths: list[str] = field(default_factory=list)
    retry_chunks: list[dict] = field(default_factory=list)

    @classmethod
    def start(cls, requests):
        progress = cls()
        for request in requests:
            key = chunk_key(request)
            progress.outcomes[key] = "unstarted"
            progress.records.setdefault((*source_key(request), request.record_id), set()).add(key)
        return progress

    def finish(self, request, outcome):
        key = chunk_key(request)
        if self.outcomes[key] != "unstarted":
            raise ValueError("a chunk was processed more than once in one invocation")
        self.outcomes[key] = outcome

    def counts(self):
        chunks = Counter(self.outcomes.values())
        records = Counter()
        for expected in self.records.values():
            states = {self.outcomes[key] for key in expected}
            if states <= {"verified", "reused"}:
                records["complete"] += 1
            elif states == {"unstarted"}:
                records["unstarted"] += 1
            else:
                records["partial"] += 1
        return {
            "chunks_processed": len(self.outcomes) - chunks["unstarted"],
            "chunks_newly_verified": chunks["verified"],
            "chunks_reused": chunks["reused"],
            "chunks_failed_this_run": chunks["failed"],
            "chunks_investigation": chunks["investigation"],
            "chunks_deferred": chunks["deferred"],
            "chunks_retry_attempted": self.retry_attempts,
            "records_complete": records["complete"],
            "records_partial": records["partial"],
            "records_unstarted": records["unstarted"],
        }

    def postfix(self):
        counts = self.counts()
        return {name: counts[name] for name in (
            "chunks_newly_verified", "chunks_reused", "chunks_failed_this_run", "chunks_deferred",
            "chunks_investigation", "records_complete", "records_partial", "records_unstarted",
        )}

    def capture_queues(self, requests, retries):
        snapshots = {}
        for request in requests:
            key = source_key(request)
            if key not in snapshots:
                queue = retries.for_request(request)
                snapshots[key] = queue.snapshot()
                self.queue_paths.append(str(queue.path))
            entry = snapshots[key].chunks[request.chunk_id]
            self.retry_chunks.append(entry_summary(entry))
        self.queue_paths = sorted(set(self.queue_paths))
        self.retry_chunks.sort(key=lambda row: (row["source_sha256"], row["source_split"], row["chunk_id"]))

    def to_dict(self):
        return {**self.counts(), "queue_paths": self.queue_paths, "retry_chunks": self.retry_chunks,
                "chunks_unresolved": sum(row["retry_status"] != "resolved" for row in self.retry_chunks),
                "chunks_dead_lettered": sum(row["retry_status"] == "dead_letter" for row in self.retry_chunks)}


def entry_summary(entry):
    return {
        "chunk_id": entry.request.chunk_id, "record_id": entry.request.record_id,
        "source_sha256": entry.request.provenance.source_sha256,
        "source_split": entry.request.provenance.source_split,
        "retry_status": entry.status,
        "attempts": entry.legacy_attempts + len(entry.history), "cycle": len(entry.cycles),
        "cycle_attempts": entry.cycle_attempts, "max_attempts": entry.cycles[-1].policy.max_attempts,
        "backoff_seconds": list(entry.cycles[-1].policy.delays_seconds),
        "last_error_category": entry.last_error_category, "terminal_reason": entry.terminal_reason,
        "next_attempt_at": entry.next_attempt_at,
    }
