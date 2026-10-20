"""Opt-in Neo4j + Ollama evaluation checks. Never contacts OpenRouter.

Creates UUID-scoped synthetic source data through the atomic ingestion boundary,
uses an isolated three-dimensional vector index, and removes its fixtures.
Extraction/ingestion embeddings are deterministic; answers and RAGAS are live.
"""

import asyncio
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.indexes import create_vector_index
from neo4j_graphrag.embeddings.base import Embedder

from graphrag_pipeline.answer import build_answer_llm
from graphrag_pipeline.config import Neo4jSettings, OllamaSettings
from graphrag_pipeline.evaluation.providers import EvaluatorSettings, build_metrics
from graphrag_pipeline.evaluation.records import DEFAULT_DATASET, load_cases
from graphrag_pipeline.evaluation.runner import run_evaluation
from graphrag_pipeline.evaluation.runtime import EvaluationSettings, PairedRetriever, graph_readiness, graph_snapshot
from graphrag_pipeline.extraction.atomic_writer import AtomicKGWriter
from graphrag_pipeline.extraction.idempotency import ParagraphSplitter, quote_identifier
from graphrag_pipeline.extraction.ingest import ingest_paragraph
from graphrag_pipeline.retrieval_prerequisites import VectorIndexSettings
from graphrag_pipeline.storage.neo4j_connection import Neo4jConnection
from test_stock_writer_behavior import FixedGraphLLM, _snapshot


class FixtureEmbeddings(Embedder):
    def __init__(self, texts):
        super().__init__()
        self.texts = texts
        self.calls = 0

    def embed_query(self, text):
        self.calls += 1
        return [1., .1 * (self.texts.index(text) + 1), 0.]


def test_live_atomic_reuse_retrieval_answers_and_ragas(tmp_path):
    if os.getenv("RUN_EVALUATION_LIVE") != "1":
        pytest.skip("Opt in with RUN_EVALUATION_LIVE=1; uses local Neo4j and Ollama only.")
    settings = Neo4jSettings.from_env()
    models = OllamaSettings.from_env()
    raw = json.loads(Path("tests/fixtures/musique/valid_records.jsonl").read_text().splitlines()[0])
    raw["id"] = "eval-live-" + uuid4().hex
    source = tmp_path / DEFAULT_DATASET.name
    source.write_text(json.dumps(raw) + "\n")
    case = load_cases(source)[0]
    index_name = "eval_live_" + uuid4().hex
    index_label = "EvalFixture_" + uuid4().hex
    requests = case.extraction_requests()
    artifact = Path(os.getenv("EVALUATION_EVIDENCE_DIR", str(tmp_path))) / index_name
    with Neo4jConnection(settings) as connection:
        connection.verify_connectivity()
        before = _snapshot(connection)
        constraints = {r["name"] for r in connection.run_read("SHOW CONSTRAINTS YIELD name RETURN name")}
        try:
            assert not graph_readiness(connection, [case])["ready"]
            extractor = FixedGraphLLM(raw["id"])
            embedder = FixtureEmbeddings([r.text for r in requests])
            pipeline = SimpleKGPipeline(
                driver=connection.connect(), llm=extractor, embedder=embedder, schema="FREE",
                from_file=False, on_error="RAISE", perform_entity_resolution=False,
                text_splitter=ParagraphSplitter(), kg_writer=AtomicKGWriter(connection.connect(), settings),
                neo4j_database=settings.database,
            )
            async def prepare():
                for request in requests:
                    assert not (await ingest_paragraph(connection, pipeline, request)).reused
                calls = (extractor.calls, embedder.calls)
                for request in requests:
                    assert (await ingest_paragraph(connection, pipeline, request)).reused
                assert (extractor.calls, embedder.calls) == calls
            asyncio.run(prepare())
            assert graph_readiness(connection, [case])["ready"]
            # A separate fixture label avoids colliding with the user's existing
            # Chunk.embedding index. Source completeness/replay was checked above.
            connection.run_write(f"MATCH (c:Chunk) WHERE c.record_id=$id SET c:{index_label}", {"id": raw["id"]})
            create_vector_index(connection.connect(), index_name, index_label, "embedding", 3, "cosine",
                fail_if_exists=True, filterable_properties=["record_id"], neo4j_database=settings.database)
            for _ in range(100):
                rows = connection.run_read("SHOW VECTOR INDEXES YIELD name, state WHERE name=$name RETURN state", {"name": index_name})
                if rows and rows[0]["state"] == "ONLINE":
                    break
                time.sleep(.1)
            else:
                pytest.fail("fixture vector index did not become online")
            config = EvaluationSettings(top_k=4, max_context_chunks=4, max_context_tokens=4096)
            retriever = PairedRetriever(connection.connect(), settings.database, models, config,
                VectorIndexSettings(name=index_name, label=index_label, dimensions=3))
            # The fixture vectors are deterministic; answer/evaluator embeddings
            # still use the user's actual Ollama models where required.
            retriever.embed = lambda question: [1., 0., 0.]
            evaluator = EvaluatorSettings("ollama", models.llm_model, models.embedding_model,
                models.host.rstrip("/") + "/v1", "ollama", retries=1, timeout=180)
            async def evaluate():
                metrics, client = build_metrics(evaluator)
                try:
                    return await run_evaluation([case], retriever, build_answer_llm(models), metrics,
                        config, artifact, snapshot=lambda: graph_snapshot(connection, settings.database), metric_timeout=180)
                finally:
                    await client.close()
            rows, summary = asyncio.run(evaluate())
            assert summary["snapshot_consistent"] and len(rows) == 4
            stored = [json.loads(s) for s in (artifact / "records.jsonl").read_text().splitlines()]
            assert all(r["response"]["status"] == "succeeded" for r in stored)
            assert all(r["paired_seeds_consistent"] and r["gold"]["answer_correct"] for r in stored)
            assert all(all(m["status"] == "scored" for m in r["ragas"].values()) for r in stored)
            assert all(r["retrieval_gold"]["all_hops_covered"] for r in stored)
            assert any(e["graph_nodes"] for r in stored if r["backend"] == "graphrag" for e in r["retrieval_evidence"])
            assert not any(e["graph_nodes"] for r in stored if r["backend"] == "vectorrag" for e in r["retrieval_evidence"])
        finally:
            connection.run_write("DROP INDEX $name IF EXISTS", {"name": index_name})
            ids = [r.chunk_id for r in requests]
            connection.run_write("MATCH (n) WHERE n.ingestion_chunk_id IN $ids OR (n:IngestionCompletion AND n.chunk_id IN $ids) DETACH DELETE n", {"ids": ids})
            current = {r["name"] for r in connection.run_read("SHOW CONSTRAINTS YIELD name RETURN name")}
            for name in current - constraints:
                if name.startswith("ingestion_"):
                    connection.run_write(f"DROP CONSTRAINT {quote_identifier(name)}")
            assert before == _snapshot(connection), "live fixture cleanup changed pre-existing graph data"
