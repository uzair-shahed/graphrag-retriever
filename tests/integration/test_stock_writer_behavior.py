"""Task 13.1: characterize the locked stock writer, not the future replacement.

Opt in with RUN_TASK13_STOCK_WRITER_TESTS=1. Only the LLM and embedder are
doubles: splitting, extraction orchestration, lexical graph construction and
Neo4j writes use the installed package. Run sequentially against a quiet DB.
TASK13_EVIDENCE_DIR optionally retains sanitized JSON reports outside pytest's
temporary directory. No provider credentials or provider requests are needed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from neo4j.exceptions import ClientError
from neo4j_graphrag.components.kg_writer import Neo4jWriter
from neo4j_graphrag.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.components.types import Neo4jGraph
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm.base import LLMInterface
from neo4j_graphrag.llm.types import LLMResponse

from graphrag_pipeline.config import Neo4jSettings
from graphrag_pipeline.extraction.plugin_input import build_plugin_input
from graphrag_pipeline.extraction.schemas import ExtractionRequest, Provenance
from graphrag_pipeline.storage.neo4j_connection import Neo4jConnection


class FixedGraphLLM(LLMInterface):
    """Return identical, valid extraction JSON on every invocation."""

    def __init__(self, run_id: str) -> None:
        super().__init__(model_name="task13-fixed-graph")
        self.calls = 0
        self.graph = Neo4jGraph.model_validate({
            "nodes": [
                {"id": node_id, "label": label, "properties": {
                    "name": name, "fixture_run_id": run_id,
                }}
                for node_id, label, name in (
                    ("person-1", "Task13Person", f"Alex {run_id}"),
                    ("person-2", "Task13Person", f"Alex {run_id}"),
                    ("company", "Task13Company", f"Aster {run_id}"),
                )
            ],
            "relationships": [
                {"start_node_id": person, "end_node_id": "company", "type": "FOUNDED"}
                for person in ("person-1", "person-2")
            ],
        })

    def invoke(self, input: str, **kwargs: Any) -> LLMResponse:
        self.calls += 1
        return LLMResponse(content=self.graph.model_dump_json())

    async def ainvoke(self, input: str, **kwargs: Any) -> LLMResponse:
        return self.invoke(input, **kwargs)


class FixedEmbedder(Embedder):
    vector = [0.25, 0.5, 0.75]

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        return list(self.vector)


def _snapshot(connection: Neo4jConnection) -> str:
    """Hash all node/relationship identities and properties without exporting them."""
    nodes = connection.run_read(
        "MATCH (n) RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS properties"
    )
    for node in nodes:
        node["labels"].sort()
    relationships = connection.run_read(
        "MATCH (a)-[r]->(b) RETURN elementId(r) AS id, type(r) AS type, "
        "elementId(a) AS source, elementId(b) AS target, properties(r) AS properties"
    )
    payload = json.dumps(
        [sorted(nodes, key=lambda row: row["id"]), sorted(relationships, key=lambda row: row["id"])],
        sort_keys=True, default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class StockFixture:
    connection: Neo4jConnection
    request: ExtractionRequest
    pipeline: SimpleKGPipeline
    llm: FixedGraphLLM
    embedder: FixedEmbedder
    report: dict[str, Any] = field(default_factory=dict)

    def inventory(self) -> dict[str, Any]:
        parameters = {"run_id": self.request.record_id, "text": self.request.text}
        groups = {
            "documents": "MATCH (n:Document {fixture_run_id: $run_id})",
            # Chunk does not receive Document metadata from the stock writer.
            "chunks": "MATCH (n:Chunk {text: $text})",
            "entities": "MATCH (n:__Entity__ {fixture_run_id: $run_id})",
        }
        result: dict[str, Any] = {}
        for name, match in groups.items():
            result[name] = self.connection.run_read(
                match + " RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS properties",
                parameters,
            )
        for name, match in {
            "source_links": "MATCH (c:Chunk {text: $text})-[r:FROM_DOCUMENT]->(d:Document {fixture_run_id: $run_id})",
            "entity_source_links": "MATCH (e:__Entity__ {fixture_run_id: $run_id})-[r:FROM_CHUNK]->(c:Chunk {text: $text})",
            "domain_relationships": "MATCH (a:__Entity__ {fixture_run_id: $run_id})-[r:FOUNDED]->(b:__Entity__ {fixture_run_id: $run_id})",
        }.items():
            result[name] = self.connection.run_read(
                match + " RETURN elementId(r) AS id, elementId(startNode(r)) AS source, "
                "elementId(endNode(r)) AS target, type(r) AS type", parameters,
            )
        result["counts"] = {name: len(rows) for name, rows in result.items()}
        result["counts"]["embeddings"] = sum(
            "embedding" in row["properties"] for row in result["chunks"]
        )
        return result

    def run(self) -> Any:
        text, metadata = build_plugin_input(self.request)
        return asyncio.run(self.pipeline.run_async(
            text=text, document_metadata={**metadata, "fixture_run_id": self.request.record_id},
        ))


@pytest.fixture
def stock_fixture(tmp_path: Path, request: pytest.FixtureRequest):
    if os.environ.get("RUN_TASK13_STOCK_WRITER_TESTS") != "1":
        pytest.skip("Opt in with RUN_TASK13_STOCK_WRITER_TESTS=1; writes isolated Neo4j fixtures.")
    # Once explicitly enabled, configuration/connectivity failures are failures,
    # not successful-looking skipped acceptance checks.
    settings = Neo4jSettings.from_env()
    if settings.database != "neo4j":
        pytest.fail("Stock writer setup/cleanup uses the default database; this fixture requires neo4j.")
    connection = Neo4jConnection(settings)
    try:
        connection.verify_connectivity()
        # Default stock cleanup clears this property globally. Do not run if
        # another ingestion's temporary state could be affected.
        temporary = connection.run_read(
            "MATCH (n:__KGBuilder__) WHERE n.__tmp_internal_id IS NOT NULL RETURN count(n) AS count"
        )
        assert temporary == [{"count": 0}], "Existing writer temporary state; run against a quiet database."
        baseline = _snapshot(connection)
        index_existed = bool(connection.run_read(
            "SHOW INDEXES YIELD name WHERE name = '__entity__tmp_internal_id' RETURN name"
        ))
        run_id = f"task13-stock-{uuid4().hex}"
        paragraph = ExtractionRequest(
            chunk_id=f"musique:{run_id}:paragraph:0", record_id=run_id,
            document_id=f"musique:{run_id}:document", paragraph_idx=0,
            text=f"Two people named Alex founded Aster. Fixture {run_id}.", title="Aster",
            provenance=Provenance(source_dataset="musique", source_split="task13_fixture",
                                  source_file="synthetic-task13.jsonl", source_sha256="a" * 64, source_line=1),
        )
        llm, embedder = FixedGraphLLM(run_id), FixedEmbedder()
        pipeline = SimpleKGPipeline(
            llm=llm, embedder=embedder, driver=connection.connect(), schema="FREE",
            text_splitter=FixedSizeSplitter(chunk_size=10000, chunk_overlap=0),
            from_file=False, on_error="RAISE", perform_entity_resolution=False,
            neo4j_database=settings.database,
        )
        fixture = StockFixture(connection, paragraph, pipeline, llm, embedder)
        fixture.report.update({
            "test": request.node.name, "fixture_run_id": run_id,
            "neo4j_graphrag_version": version("neo4j-graphrag"),
            "neo4j_driver_version": version("neo4j"),
            "server": connection.run_read(
                "CALL dbms.components() YIELD versions, edition RETURN versions, edition"
            ),
            "perform_entity_resolution": pipeline.runner.config.perform_entity_resolution,
            "baseline_graph_sha256": baseline,
        })
        try:
            yield fixture
        finally:
            # Exact run UUID and paragraph text locate even nodes from a
            # failed write that never created FROM_DOCUMENT relationships.
            connection.run_write(
                "MATCH (n) WHERE n.fixture_run_id = $run_id OR (n:Chunk AND n.text = $text) DETACH DELETE n",
                {"run_id": run_id, "text": paragraph.text},
            )
            if not index_existed:
                connection.run_write("DROP INDEX __entity__tmp_internal_id IF EXISTS")
            after = _snapshot(connection)
            fixture.report.update({
                "llm_calls": llm.calls, "embedding_calls": embedder.calls,
                "cleanup_graph_sha256": after, "cleanup_preserved_existing_graph": after == baseline,
                "remaining_fixture_counts": fixture.inventory()["counts"],
            })
            evidence_dir = Path(os.environ.get("TASK13_EVIDENCE_DIR", str(tmp_path)))
            evidence_dir.mkdir(parents=True, exist_ok=True)
            (evidence_dir / f"{request.node.name}.json").write_text(
                json.dumps(fixture.report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
            )
            assert after == baseline, "Existing graph changed during fixture execution/cleanup."
            assert not any(fixture.report["remaining_fixture_counts"].values())
    finally:
        connection.close()


def test_stock_pipeline_replay_duplicates_graph(stock_fixture: StockFixture) -> None:
    fixture = stock_fixture
    assert fixture.pipeline.runner.config.perform_entity_resolution is False
    assert fixture.pipeline.runner.config.kg_writer is None  # installed default Neo4jWriter
    expected = {"documents": 1, "chunks": 1, "entities": 3, "source_links": 1,
                "entity_source_links": 3, "domain_relationships": 2, "embeddings": 1}
    previous_ids: dict[str, set[str]] = {}
    for attempt in (1, 2):
        result = fixture.run()
        fixture.report[f"pipeline_result_{attempt}"] = result.model_dump(mode="json")
        assert result.result["writer"]["status"] == "SUCCESS"
        inventory = fixture.inventory()
        fixture.report[f"inventory_{attempt}"] = inventory
        assert inventory["counts"] == {key: value * attempt for key, value in expected.items()}
        for group in expected.keys() - {"embeddings"}:
            ids = {row["id"] for row in inventory[group]}
            assert len(ids) == expected[group] * attempt
            assert previous_ids.get(group, set()) <= ids
            previous_ids[group] = ids
        assert all(row["properties"]["text"] == fixture.request.text for row in inventory["chunks"])
        assert all(row["properties"]["embedding"] == FixedEmbedder.vector for row in inventory["chunks"])
        _, expected_metadata = build_plugin_input(fixture.request)
        for document in inventory["documents"]:
            assert all(document["properties"].get(key) == value for key, value in expected_metadata.items())
        # Metadata lives on Document; the application's separate synchronization
        # step, not this stock writer, supplies Chunk's application chunk_id.
        assert all("chunk_id" not in row["properties"] for row in inventory["chunks"])
        document_ids = {row["id"] for row in inventory["documents"]}
        chunk_ids = {row["id"] for row in inventory["chunks"]}
        entity_ids = {row["id"] for row in inventory["entities"]}
        assert {row["source"] for row in inventory["source_links"]} == chunk_ids
        assert {row["target"] for row in inventory["source_links"]} == document_ids
        assert {row["source"] for row in inventory["entity_source_links"]} == entity_ids
        assert {row["target"] for row in inventory["entity_source_links"]} == chunk_ids
        companies = {row["id"] for row in inventory["entities"] if "Task13Company" in row["labels"]}
        people = entity_ids - companies
        assert {row["source"] for row in inventory["domain_relationships"]} == people
        assert {row["target"] for row in inventory["domain_relationships"]} == companies
        alex = [row for row in inventory["entities"] if row["properties"]["name"].startswith("Alex ")]
        assert len(alex) == 2 * attempt  # same-label/name entities remain separate
        assert fixture.llm.calls == fixture.embedder.calls == attempt


def test_stock_writer_client_error_leaves_committed_nodes(
    stock_fixture: StockFixture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = stock_fixture

    def fail_relationship_batch(self, relationships):
        raise ClientError("Task 13 injected failure after committed node batches")

    monkeypatch.setattr(Neo4jWriter, "_upsert_relationships", fail_relationship_batch)
    result = fixture.run()  # on_error=RAISE does not raise this writer failure
    inventory = fixture.inventory()
    fixture.report.update({"pipeline_result": result.model_dump(mode="json"), "inventory": inventory})
    assert result.result["writer"]["status"] == "FAILURE"
    assert "Task 13 injected failure" in result.result["writer"]["metadata"]["error"]
    assert inventory["counts"] == {
        "documents": 1, "chunks": 1, "entities": 3, "source_links": 0,
        "entity_source_links": 0, "domain_relationships": 0, "embeddings": 1,
    }
    assert fixture.llm.calls == fixture.embedder.calls == 1
