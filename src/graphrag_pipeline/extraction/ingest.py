"""Validated, durably tracked ingestion shared by batch, preload and fixtures."""

from dataclasses import dataclass
from typing import Any

from graphrag_pipeline.extraction.atomic_writer import IngestionStateError
from graphrag_pipeline.extraction.plugin_input import build_plugin_input
from graphrag_pipeline.ingestion.reconcile import ABSENT, COMPLETE, ChunkState, classify_chunk
from graphrag_pipeline.ingestion.retry_policy import classify_failure, failure_stage, FailureDecision
from graphrag_pipeline.ingestion.progress_manifest import ManifestError
from graphrag_pipeline.ingestion.retry_queue import QueueError, RetryQueue, RetryStore


class RetryDeferred(RuntimeError):
    """Chunk remains unresolved, without authorizing another pipeline attempt."""


@dataclass
class IngestionResult:
    state: ChunkState
    result: Any = None
    reused: bool = False


async def ingest_paragraph(connection, pipeline, request, *, before_write=None, classify=None,
                           retry_queue: RetryQueue | None = None, configuration=None, pipeline_factory=None) -> IngestionResult:
    """Reconcile before model calls and resolve the queue before checkpoints.

    The pipeline may be a lazy factory, letting reconciliation skip construction
    for committed work. Queue errors and local checkpoint failures propagate;
    they are never misclassified as model failures. Abrupt termination leaves
    an in-flight journal entry until authoritative Neo4j reconciliation.
    """
    classify = classify if classify is not None else classify_chunk
    queue = retry_queue if retry_queue is not None else RetryStore().for_request(request)
    state = reconcile_paragraph(connection, request, queue, classify=classify)
    if state.state == COMPLETE:
        return IngestionResult(state, reused=True)
    if state.state != ABSENT:
        return IngestionResult(state)
    if request.chunk_id not in queue.due_ids():
        entry = queue.snapshot().chunks[request.chunk_id]
        raise RetryDeferred(f"{entry.status}: {entry.last_error_category}; next_attempt_at={entry.next_attempt_at}")
    # Construction errors are distinct from extraction output validation.
    try:
        active_pipeline = pipeline_factory() if pipeline_factory is not None else pipeline
    except QueueError:
        raise
    except Exception as error:
        queue.record_observation(request.chunk_id, classify_failure(error, "configuration"), "configuration")
        raise
    try:
        text, metadata = build_plugin_input(request)
    except Exception as error:
        queue.record_observation(request.chunk_id, classify_failure(error, "input"), "input")
        raise
    if before_write is not None:
        try:
            before_write()
        except (ManifestError, QueueError):
            raise
        except Exception as error:
            raise ManifestError("pre-write progress checkpoint failed") from error
    queue.begin_attempt(request.chunk_id, configuration or {})
    try:
        result = await active_pipeline.run_async(text=text, document_metadata=metadata)
    except QueueError:
        raise
    except Exception as error:
        stage = failure_stage(error)
        queue.fail_attempt(request.chunk_id, classify_failure(error, stage), stage)
        raise
    try:
        written = classify(connection, request)
        if written.state != COMPLETE:
            raise IngestionStateError(f"post-write Neo4j verification failed ({written.state}): {written.reason}")
    except Exception as error:
        queue.fail_attempt(request.chunk_id, classify_failure(error, "reconcile"), "reconcile")
        raise
    queue.resolve_verified(request.chunk_id)
    return IngestionResult(written, result=result)


def reconcile_paragraph(connection, request, queue, *, classify=None):
    """Read authoritative graph state and update the journal without provider work."""
    classify = classify if classify is not None else classify_chunk
    queue.ensure_request(request)
    try:
        state = classify(connection, request)
    except Exception as error:
        queue.record_observation(request.chunk_id, classify_failure(error, "reconcile"), "reconcile")
        raise
    if state.state == COMPLETE:
        queue.resolve_verified(request.chunk_id)
        return state
    if state.state != ABSENT:
        queue.record_observation(request.chunk_id, FailureDecision("unsafe_graph_state", False), "reconcile")
        return state
    entry = queue.snapshot().chunks[request.chunk_id]
    if entry.status == "in_flight":
        queue.reconcile_absent(request.chunk_id)
    elif entry.status == "resolved":
        queue.record_observation(request.chunk_id, FailureDecision("verified_graph_missing", True), "reconcile")
    return state
