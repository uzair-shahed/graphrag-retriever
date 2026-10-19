"""Subprocess driver for real Task 13 crash/restart acceptance tests.

Use only from test_atomic_ingestion.py. The database and ingestion code are real;
provider constructors are replaced with deterministic test implementations.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
from unittest.mock import patch

from graphrag_pipeline.config import Neo4jSettings, OllamaSettings
from graphrag_pipeline.extraction.atomic_writer import AtomicKGWriter
from graphrag_pipeline.extraction.schemas import ExtractionRequest
from graphrag_pipeline.ingestion.load_musique_dataset import RunOutcome, run_requests
from graphrag_pipeline.ingestion.progress_manifest import ManifestStore
from test_stock_writer_behavior import FixedEmbedder, FixedGraphLLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("events", type=Path)
    parser.add_argument("--stop", choices=["before_commit", "after_commit"])
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    request = ExtractionRequest.model_validate_json(args.request.read_text())

    def event(kind):
        with args.events.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"kind": kind, "chunk_id": request.chunk_id}) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    llm, embedder = FixedGraphLLM(request.record_id), FixedEmbedder()
    real_invoke, real_embed = llm.invoke, embedder.embed_query

    def invoke(*args, **kwargs):
        event("llm")
        return real_invoke(*args, **kwargs)

    def embed(*args, **kwargs):
        event("embedding")
        return real_embed(*args, **kwargs)

    llm.invoke, embedder.embed_query = invoke, embed
    original = AtomicKGWriter._write

    def write(self, connection, *write_args):
        result = original(self, connection, *write_args)
        if args.stop == "before_commit":
            event("before_commit")
            os._exit(74)  # real process death with the complete transaction uncommitted
        return result

    # Stop immediately before the checkpoint, after central ingestion has
    # independently verified that the database commit is complete.
    from graphrag_pipeline.ingestion.progress_manifest import ProgressManifest
    original_verified = ProgressManifest.mark_verified

    def verified(self, chunk_id):
        if args.stop == "after_commit":
            event("after_commit_before_checkpoint")
            os._exit(73)
        return original_verified(self, chunk_id)

    with (
        patch("graphrag_pipeline.extraction.pipeline.OllamaJSONSchemaLLM", return_value=llm),
        patch("graphrag_pipeline.extraction.pipeline.OllamaEmbeddings", return_value=embedder),
        patch.object(AtomicKGWriter, "_write", write),
        patch.object(ProgressManifest, "mark_verified", verified),
    ):
        store = ManifestStore(manifest_path=args.manifest, fresh=args.fresh)
        outcome = RunOutcome()
        succeeded, failed = asyncio.run(run_requests(
            [request], Neo4jSettings.from_env(), OllamaSettings(host="http://unused", llm_model="fixed", embedding_model="fixed"),
            show_progress=False, store=store, outcome=outcome,
        ))
        report = {"succeeded": succeeded, "failed": failed, "outcome": outcome.to_dict(),
                  "complete_records": sorted(store.complete_record_ids([request]))}
        print(json.dumps(report))
        raise SystemExit(1 if failed or outcome.needs_investigation else 0)


if __name__ == "__main__":
    main()
