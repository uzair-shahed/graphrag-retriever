"""Real-process retry acceptance driver; providers are deterministic test doubles."""

import argparse
import asyncio
import json
import os
from pathlib import Path
from unittest.mock import patch

from graphrag_pipeline.config import Neo4jSettings, OllamaSettings
from graphrag_pipeline.extraction.atomic_writer import AtomicKGWriter
from graphrag_pipeline.extraction.pipeline import build_simple_kg_pipeline
from graphrag_pipeline.extraction.schemas import ExtractionRequest
from graphrag_pipeline.ingestion import load_musique_dataset as loader
from graphrag_pipeline.ingestion import retry_chunks as retry_cli
from graphrag_pipeline.ingestion.progress_manifest import ManifestStore
from graphrag_pipeline.ingestion.retry_queue import RetryQueue
from test_stock_writer_behavior import FixedGraphLLM, FixedEmbedder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("requests", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("events", type=Path)
    parser.add_argument("--operation", choices=["batch", "inspect", "drain", "requeue"], default="batch")
    parser.add_argument("--queue-path", type=Path)
    parser.add_argument("--chunk-id", action="append", default=[])
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--reason", default="reviewed and repaired injected failure")
    parser.add_argument("--target")
    parser.add_argument("--fail", action="store_true")
    parser.add_argument("--fail-all", action="store_true")
    parser.add_argument("--pause", action="store_true")
    parser.add_argument("--stop", choices=["before_commit", "after_commit", "after_failure"])
    args = parser.parse_args()
    if args.operation == "batch":
        requests = [ExtractionRequest.model_validate(value) for value in json.loads(args.requests.read_text())]
    else:
        requests = [e.request for e in RetryQueue.load(args.queue_path).snapshot().chunks.values()]
    llm, embedder = FixedGraphLLM(requests[0].record_id), FixedEmbedder()
    models = OllamaSettings(host="http://unused", llm_model="fixed", embedding_model="fixed")
    current = {"chunk_id": None}

    def event(kind):
        with args.events.open("a") as handle:
            handle.write(json.dumps({"kind": kind, "chunk_id": current["chunk_id"]}) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    invoke, embed = llm.invoke, embedder.embed_query

    def tracked_invoke(*a, **kw):
        event("llm")
        return invoke(*a, **kw)

    def tracked_embed(*a, **kw):
        event("embedding")
        return embed(*a, **kw)

    llm.invoke, embedder.embed_query = tracked_invoke, tracked_embed

    def build(*a, **kw):
        pipeline = build_simple_kg_pipeline(*a, **kw)
        run = pipeline.run_async

        async def tracked_run(**kwargs):
            current["chunk_id"] = kwargs["document_metadata"]["chunk_id"]
            if args.pause and current["chunk_id"] == args.target:
                (args.events.parent / "ready").write_text("in flight")
                while not (args.events.parent / "release").exists():
                    await asyncio.sleep(0.01)
            return await run(**kwargs)

        pipeline.run_async = tracked_run
        return pipeline

    write, resolve, fail = AtomicKGWriter._write, RetryQueue.resolve_verified, RetryQueue.fail_attempt

    def intercepted_write(self, connection, request, *a):
        result = write(self, connection, request, *a)
        if request.chunk_id == args.target and args.stop == "before_commit":
            os._exit(74)
        if args.fail_all or request.chunk_id == args.target and args.fail:
            raise TimeoutError("injected retry acceptance failure before commit")
        return result

    def intercepted_resolve(self, chunk_id):
        if chunk_id == args.target and args.stop == "after_commit":
            os._exit(73)
        return resolve(self, chunk_id)

    def intercepted_fail(self, chunk_id, *a):
        result = fail(self, chunk_id, *a)
        if chunk_id == args.target and args.stop == "after_failure":
            os._exit(75)
        return result

    with patch("graphrag_pipeline.extraction.pipeline.OllamaJSONSchemaLLM", return_value=llm), patch("graphrag_pipeline.extraction.pipeline.OllamaEmbeddings", return_value=embedder), patch.object(loader, "build_simple_kg_pipeline", side_effect=build), patch.object(retry_cli, "build_simple_kg_pipeline", side_effect=build), patch.object(retry_cli, "model_settings_from_env", return_value=models), patch.object(AtomicKGWriter, "_write", intercepted_write), patch.object(RetryQueue, "resolve_verified", intercepted_resolve), patch.object(RetryQueue, "fail_attempt", intercepted_fail):
        if args.operation != "batch":
            argv = [args.operation, "--queue-path", str(args.queue_path)]
            for cid in args.chunk_id:
                argv.extend(["--chunk-id", cid])
            if args.operation == "requeue":
                argv.extend(["--reason", args.reason])
            if args.operation == "drain" and args.max_chunks is not None:
                argv.extend(["--max-chunks", str(args.max_chunks)])
            raise SystemExit(retry_cli.run_cli(argv))
        outcome = loader.RunOutcome()
        store = ManifestStore(manifest_path=args.manifest)
        succeeded, failed = asyncio.run(loader.run_requests(requests, Neo4jSettings.from_env(), models, store=store, outcome=outcome, show_progress=False))
        report = {"succeeded": succeeded, "failed": failed, "outcome": outcome.to_dict()}
        print(json.dumps(report))
        raise SystemExit(1 if failed or outcome.progress.to_dict()["chunks_unresolved"] else 0)


if __name__ == "__main__":
    main()
