"""Inspect, drain or explicitly requeue a source's durable chunk retry journal."""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
from pathlib import Path

from graphrag_pipeline.config import Neo4jSettings, model_settings_from_env
from graphrag_pipeline.diagnostics import exception_diagnostics
from graphrag_pipeline.extraction.ingest import ingest_paragraph, reconcile_paragraph, RetryDeferred
from graphrag_pipeline.extraction.pipeline import build_simple_kg_pipeline
from graphrag_pipeline.ingestion.progress_manifest import ProgressManifest, ManifestError, default_manifest_path
from graphrag_pipeline.ingestion.reconcile import COMPLETE, ABSENT
from graphrag_pipeline.ingestion.retry_policy import RetryPolicy, attempt_configuration, classify_failure
from graphrag_pipeline.ingestion.retry_queue import RetryQueue, QueueError
from graphrag_pipeline.ingestion.run_progress import entry_summary
from graphrag_pipeline.storage.neo4j_connection import Neo4jConnection


def select_ids(queue, record_ids=(), chunk_ids=()):
    data = queue.snapshot()
    missing_records = set(record_ids) - set(data.records)
    missing_chunks = set(chunk_ids) - set(data.chunks)
    if missing_records or missing_chunks:
        raise QueueError(f"unknown queue selection: records={sorted(missing_records)}, chunks={sorted(missing_chunks)}")
    selected = sorted(cid for cid, entry in data.chunks.items()
                      if (not record_ids or entry.request.record_id in record_ids)
                      and (not chunk_ids or cid in chunk_ids))
    if (record_ids or chunk_ids) and not selected:
        raise QueueError("record and chunk filters do not intersect")
    return selected


def inspect_queue(queue, selected):
    data, due = queue.snapshot(), set(queue.due_ids())
    return {
        "queue_path": str(queue.path), "source_sha256": data.source_sha256,
        "source_split": data.source_split, "manifest_paths": data.manifest_paths,
        "authority": "local_queue_state; Neo4j_not_checked",
        "entries": [{**entry_summary(data.chunks[cid]), "due": cid in due,
                     "provenance": data.chunks[cid].request.provenance.model_dump(mode="json"),
                     "history": [a.model_dump(mode="json") for a in data.chunks[cid].history],
                     "observations": [o.model_dump(mode="json") for o in data.chunks[cid].observations],
                     "cycles": [c.model_dump(mode="json") for c in data.chunks[cid].cycles]}
                    for cid in selected],
    }


def linked_manifests(queue, record_ids, override=None):
    data = queue.snapshot()
    paths = set(data.manifest_paths)
    fallback = default_manifest_path(data.source_sha256, data.source_split).resolve()
    if fallback.exists():
        paths.add(str(fallback))
    if override is not None:
        if not override.exists():
            raise ManifestError(f"explicit progress manifest does not exist: {override}")
        paths.add(str(override.resolve()))
    manifests, missing = [], []
    for value in sorted(paths):
        path = Path(value)
        if not path.exists():
            missing.append(value)
            continue
        manifest = ProgressManifest.open(path, data.source_sha256, data.source_split)
        for record_id in record_ids:
            known = manifest._data["records"].get(record_id)
            if known is not None and set(known["expected_chunk_ids"]) != set(data.records[record_id]):
                raise ManifestError(f"queue and manifest disagree on expected chunks for {record_id}")
        manifests.append(manifest)
    if override is not None:
        queue.remember_manifest(override)
    return manifests, missing


def checkpoint(manifests, queue, request, state=None, *, before=False):
    entry = queue.snapshot().chunks[request.chunk_id]
    for manifest in manifests:
        if request.chunk_id not in manifest._data["chunks"]:
            continue
        if before:
            manifest.mark_in_flight(request.chunk_id)
        elif state is not None and state.state == COMPLETE:
            manifest.mark_verified(request.chunk_id)
        elif state is not None and state.state != ABSENT:
            manifest.mark_needs_investigation(request.chunk_id, state.to_report(request))
        elif entry.status == "pending":
            manifest.require_verification(request.chunk_id)
        else:
            manifest.mark_failed(request.chunk_id, entry_summary(entry))


async def drain_queue(queue, selected, *, max_chunks=None, manifest_path=None, stop_requested=None,
                      neo4j_settings=None, model_settings=None):
    """One deterministic candidate snapshot; siblings receive read-only checks.

    No dataset loads or waits. The provider-work limit excludes sibling checks,
    which are necessary before certifying record completion. Only selected chunks
    may run providers, even if an unselected sibling is missing.
    """
    initial = queue.snapshot()
    affected = sorted({initial.chunks[cid].request.record_id for cid in selected})
    manifests, missing = linked_manifests(queue, affected, manifest_path)
    eligible = set(queue.due_ids()) | {cid for cid, entry in initial.chunks.items() if entry.status == "in_flight"}
    candidates = [cid for cid in selected if cid in eligible]
    if max_chunks is not None:
        candidates = candidates[:max_chunks]
    processed, errors, record_results = [], [], []
    stopped, pipeline = False, None
    settings = neo4j_settings
    connection = None
    models = model_settings

    def get_pipeline():
        nonlocal pipeline, models
        if pipeline is None:
            models = models or model_settings_from_env()
            pipeline = build_simple_kg_pipeline(connection.connect(), models, neo4j_settings=settings)
        return pipeline

    try:
        if selected:
            try:
                settings = settings or Neo4jSettings.from_env()
            except Exception as error:
                for cid in selected:
                    queue.record_observation(cid, classify_failure(error, "configuration"), "configuration")
                raise
            connection = Neo4jConnection(settings)
        for cid in candidates:
            if stop_requested is not None and stop_requested():
                stopped = True
                break
            request = initial.chunks[cid].request
            before = queue.snapshot().chunks[cid]
            status = "failed"
            try:
                # Model settings are read lazily by the pipeline factory. Capture
                # them after construction, before the actual provider attempt.
                configuration = {}

                def configured_pipeline():
                    active = get_pipeline()
                    configuration.update(attempt_configuration(settings, models))
                    return active

                result = await ingest_paragraph(connection, None, request, retry_queue=queue,
                    pipeline_factory=configured_pipeline, configuration=configuration,
                    before_write=lambda: checkpoint(manifests, queue, request, before=True))
                status = "reused" if result.reused else "verified" if result.state.state == COMPLETE else "investigation"
                checkpoint(manifests, queue, request, result.state)
            except (QueueError, ManifestError):
                raise
            except RetryDeferred:
                status = "deferred"
                checkpoint(manifests, queue, request)
            except Exception as error:
                errors.append({"chunk_id": cid, **exception_diagnostics(error)})
                checkpoint(manifests, queue, request)
            after = queue.snapshot().chunks[cid]
            processed.append({"chunk_id": cid, "outcome": status,
                              "attempt_started": len(after.history) > len(before.history),
                              "retry_attempt": len(after.history) > len(before.history) and bool(before.history or before.legacy_attempts)})
        # Queue status alone cannot prove that a parent record is complete.
        # Independently reconcile every expected sibling, without running models.
        for record_id in affected:
            verified = []
            for cid in initial.records[record_id]:
                request = initial.chunks[cid].request
                try:
                    state = reconcile_paragraph(connection, request, queue)
                    checkpoint(manifests, queue, request, state)
                    if state.state == COMPLETE:
                        verified.append(cid)
                except (QueueError, ManifestError):
                    raise
                except Exception as error:
                    errors.append({"chunk_id": cid, "stage": "record_verification", **exception_diagnostics(error)})
                    checkpoint(manifests, queue, request)
            record_results.append({"record_id": record_id, "expected_chunks": len(initial.records[record_id]),
                                   "verified_chunks": len(verified), "complete": len(verified) == len(initial.records[record_id])})
    finally:
        if connection is not None:
            connection.close()
    final = queue.snapshot()
    unresolved = [cid for cid in selected if final.chunks[cid].status != "resolved"]
    return {
        "queue_path": str(queue.path), "chunks_selected": len(selected), "chunks_processed": len(processed),
        "chunks_newly_verified": sum(row["outcome"] == "verified" for row in processed),
        "chunks_reused": sum(row["outcome"] == "reused" for row in processed),
        "chunks_failed_this_run": sum(row["outcome"] == "failed" for row in processed),
        "chunks_retry_attempted": sum(row["retry_attempt"] for row in processed),
        "chunks_deferred": sum(final.chunks[cid].status in {"pending", "retry_wait", "in_flight"} for cid in unresolved),
        "chunks_dead_lettered": sum(final.chunks[cid].status == "dead_letter" for cid in unresolved),
        "chunks_unresolved": len(unresolved), "stopped_early": stopped,
        "records_complete": sum(row["complete"] for row in record_results),
        "records_partial": sum(not row["complete"] for row in record_results),
        "record_verification": record_results, "processed": processed, "errors": errors,
        "retry_chunks": [entry_summary(final.chunks[cid]) for cid in selected],
        "manifest_paths": [str(manifest.path) for manifest in manifests], "missing_manifest_paths": missing,
        "succeeded": not unresolved and not stopped,
    }


def _positive(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def run_cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    for operation in ("inspect", "drain", "requeue"):
        command = commands.add_parser(operation)
        command.add_argument("--queue-path", type=Path, required=True)
        command.add_argument("--record-id", action="append", default=[])
        command.add_argument("--chunk-id", action="append", default=[], required=operation == "requeue")
        if operation == "drain":
            command.add_argument("--max-chunks", type=_positive)
            command.add_argument("--manifest-path", type=Path)
        elif operation == "requeue":
            command.add_argument("--reason", required=True)
    args = parser.parse_args(argv)
    previous, stop = {}, {"requested": False}

    def request_stop(signum, _frame):
        stop["requested"] = True
        signal.signal(signum, signal.SIG_DFL)
        print("Stop requested: finishing the current chunk before exiting.", file=sys.stderr, flush=True)

    try:
        queue = RetryQueue.load(args.queue_path)
        selected = select_ids(queue, args.record_id, args.chunk_id)
        if args.operation == "inspect":
            report, code = inspect_queue(queue, selected), 0
        elif args.operation == "requeue":
            queue.requeue_many(selected, args.reason, RetryPolicy.from_env())
            report, code = {"requeued_chunk_ids": selected, **inspect_queue(queue, selected)}, 0
        else:
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous[signum] = signal.signal(signum, request_stop)
            report = asyncio.run(drain_queue(queue, selected, max_chunks=args.max_chunks,
                manifest_path=args.manifest_path, stop_requested=lambda: stop["requested"]))
            code = 130 if report["stopped_early"] else 0 if report["succeeded"] else 1
        print(json.dumps(report, indent=2, sort_keys=True))
        return code
    except (QueueError, ManifestError, ValueError) as error:
        print(json.dumps(exception_diagnostics(error), indent=2))
        return 2
    except Exception as error:
        print(json.dumps(exception_diagnostics(error), indent=2))
        return 1
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def main():
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()
