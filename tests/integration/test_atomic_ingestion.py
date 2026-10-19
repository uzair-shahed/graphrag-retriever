"""Real Neo4j acceptance tests for Task 13; no model services are contacted.

RUN_TASK13_ATOMIC_TESTS=1 opts in; once enabled, unavailable services fail.
Each test deletes only UUID-scoped data and constraints created during that test.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from unittest.mock import patch
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import pytest
from neo4j.exceptions import ConstraintError, ServiceUnavailable, TransientError
from neo4j_graphrag.components.types import LexicalGraphConfig, Neo4jGraph
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

from graphrag_pipeline.config import Neo4jSettings, OllamaSettings
from graphrag_pipeline.extraction.atomic_writer import AtomicKGWriter, TransactionConnection
from graphrag_pipeline.extraction.idempotency import ParagraphSplitter, ensure_schema, managed_key, read_inventory, quote_identifier
from graphrag_pipeline.extraction import idempotency
from graphrag_pipeline.extraction.ingest import ingest_paragraph
from graphrag_pipeline.extraction.plugin_input import build_plugin_input
from graphrag_pipeline.extraction.schemas import ExtractionRequest, Provenance
from graphrag_pipeline.ingestion.reconcile import ABSENT, COMPLETE, CONFLICT, LEGACY, PARTIAL, classify_chunk
from graphrag_pipeline.storage.neo4j_connection import Neo4jConnection
from graphrag_pipeline.ingestion.load_musique_dataset import collect_requests, load_dataset, load_selected_records
from graphrag_pipeline.extraction.run_fixture import run_fixture
from graphrag_pipeline.query import select_queries
from test_stock_writer_behavior import FixedEmbedder, FixedGraphLLM, _snapshot


@dataclass
class World:
    connection: Neo4jConnection
    settings: Neo4jSettings = field(repr=False)
    run_id: str
    requests: list = field(default_factory=list)
    evidence: dict = field(default_factory=dict)

    def request(self, paragraph_idx=0, **changes):
        request = ExtractionRequest(
            chunk_id=f"musique:{self.run_id}:paragraph:{paragraph_idx}", record_id=self.run_id,
            document_id=f"{self.run_id}:document", paragraph_idx=paragraph_idx,
            text=f"  Alex founded Aster. {self.run_id}, paragraph {paragraph_idx}.\n", title=None,
            provenance=Provenance(source_dataset="musique", source_split="task13_fixture",
                                  source_file=f"{self.run_id}.jsonl", source_sha256="a" * 64, source_line=1),
        ).model_copy(update=changes)
        self.requests.append(request)
        return request

    def pipeline(self, writer=None):
        llm, embedder = FixedGraphLLM(self.run_id), FixedEmbedder()
        writer = writer or AtomicKGWriter(self.connection.connect(), self.settings)
        pipeline = SimpleKGPipeline(
            driver=self.connection.connect(), llm=llm, embedder=embedder, schema="FREE",
            from_file=False, on_error="RAISE", perform_entity_resolution=False,
            text_splitter=ParagraphSplitter(), kg_writer=writer, neo4j_database=self.settings.database,
        )
        return pipeline, llm, embedder


@pytest.fixture
def world(tmp_path, request):
    if os.environ.get("RUN_TASK13_ATOMIC_TESTS") != "1":
        pytest.skip("Opt in with RUN_TASK13_ATOMIC_TESTS=1; writes UUID-scoped fixtures.")
    settings = Neo4jSettings.from_env()
    with Neo4jConnection(settings) as connection:
        connection.verify_connectivity()
        before = _snapshot(connection)
        constraints = {row["name"] for row in connection.run_read("SHOW CONSTRAINTS YIELD name RETURN name")}
        fixture = World(connection, settings, "task13-atomic-" + uuid4().hex)
        fixture.evidence["server"] = connection.run_read("CALL dbms.components() YIELD versions, edition RETURN versions, edition")
        try:
            yield fixture
        finally:
            ids = list({req.chunk_id for req in fixture.requests})
            connection.run_write(
                "MATCH (n) WHERE n.ingestion_chunk_id IN $ids OR n.fixture_run_id = $run_id "
                "OR (n:IngestionCompletion AND n.chunk_id IN $ids) "
                "OR ((n:Document OR n:Chunk) AND n.chunk_id IN $ids) DETACH DELETE n", {"ids": ids, "run_id": fixture.run_id},
            )
            current = {row["name"] for row in connection.run_read("SHOW CONSTRAINTS YIELD name RETURN name")}
            for name in current - constraints:
                if name.startswith("ingestion_"):
                    connection.run_write(f"DROP CONSTRAINT {quote_identifier(name)}")
            after = _snapshot(connection)
            fixture.evidence.update(test=request.node.name, before=before, after=after, cleanup_preserved_graph=before == after)
            directory = Path(os.environ.get("TASK13_EVIDENCE_DIR", str(tmp_path)))
            directory.mkdir(parents=True, exist_ok=True)
            (directory / (request.node.name + ".json")).write_text(json.dumps(fixture.evidence, indent=2, sort_keys=True) + "\n")
            assert before == after, "test cleanup changed existing graph data"


def test_identity_schema_and_legacy_preservation(world):
    connection, request = world.connection, world.request()
    ensure_schema(connection, ["FROM_DOCUMENT"])
    ensure_schema(connection, ["FROM_DOCUMENT"])
    _, metadata = build_plugin_input(request)
    connection.run_write("CREATE (d:Document) SET d = $metadata", {"metadata": metadata})
    state = classify_chunk(connection, request)
    assert state.state == PARTIAL  # legacy Document exists but its Chunk is missing
    connection.run_write(
        "MATCH (d:Document {chunk_id: $id}) CREATE (c:Chunk) SET c = $props "
        "CREATE (c)-[:FROM_DOCUMENT]->(d)",
        {"id": request.chunk_id, "props": {**metadata, "text": request.text,
                                           "ingestion_chunk_id": request.chunk_id}},
    )
    # ingestion_chunk_id is only fixture cleanup ownership, not a completion claim.
    legacy = _snapshot(connection)
    assert classify_chunk(connection, request).state == LEGACY
    assert _snapshot(connection) == legacy
    changed = request.model_copy(update={"text": "different"})
    assert classify_chunk(connection, changed).state == CONFLICT
    connection.run_write("CREATE (:IngestionCompletion {chunk_id: $id})", {"id": request.chunk_id})
    with pytest.raises(ConstraintError):
        connection.run_write("CREATE (:IngestionCompletion {chunk_id: $id})", {"id": request.chunk_id})
    connection.run_write("CREATE (:__IngestedNode__ {ingestion_key: $key, ingestion_chunk_id: $id})",
                         {"id": request.chunk_id, "key": request.chunk_id})
    with pytest.raises(ConstraintError):
        connection.run_write("CREATE (:__IngestedNode__ {ingestion_key: $key})", {"key": request.chunk_id})
    # Relationship uniqueness is checked by Neo4j, not just a Python preflight.
    connection.run_write(
        "MATCH (d:Document {chunk_id: $id}), (c:Chunk {ingestion_chunk_id: $id}) "
        "CREATE (c)-[:FROM_DOCUMENT {ingestion_key: $key}]->(d)", {"id": request.chunk_id, "key": request.chunk_id},
    )
    with pytest.raises(ConstraintError):
        connection.run_write(
            "MATCH (d:Document {chunk_id: $id}), (c:Chunk {ingestion_chunk_id: $id}) "
            "CREATE (c)-[:FROM_DOCUMENT {ingestion_key: $key}]->(d)", {"id": request.chunk_id, "key": request.chunk_id},
        )
    world.evidence.update(legacy_state=LEGACY, legacy_unchanged=True, uniqueness_checks=3)


def test_legacy_orphan_chunk_is_preserved_and_not_reingested(world):
    request = world.request()
    world.connection.run_write("CREATE (:Chunk {chunk_id: $id, text: $text})",
                               {"id": request.chunk_id, "text": request.text})
    before = _snapshot(world.connection)
    pipeline, llm, embedder = world.pipeline()
    result = asyncio.run(ingest_paragraph(world.connection, pipeline, request))
    assert result.state.state == PARTIAL
    assert not llm.calls and not embedder.calls
    assert _snapshot(world.connection) == before
    world.evidence.update(orphan_preserved=True, provider_calls=0)


def test_incompatible_schema_name_is_explicit(world, monkeypatch):
    ensure_schema(world.connection)
    before = _snapshot(world.connection)
    # Simulate a same-name constraint for the wrong required label, without
    # dropping or changing any installed production constraint.
    monkeypatch.setattr(idempotency, "COMPLETION_LABEL", "UnexpectedCompletionLabel")
    with pytest.raises(RuntimeError, match="incompatible ingestion constraint"):
        ensure_schema(world.connection)
    assert _snapshot(world.connection) == before
    world.evidence["incompatible_schema_rejected"] = True


def test_unattributed_managed_key_is_never_overwritten(world):
    request = world.request()
    world.connection.run_write(
        "CREATE (:__IngestedNode__ {ingestion_key: $key, fixture_run_id: $run_id, preserved: 'original'})",
        {"key": managed_key(request.chunk_id, "document", "source"), "run_id": world.run_id},
    )
    before = _snapshot(world.connection)
    pipeline, _, _ = world.pipeline()
    with pytest.raises(Exception, match="managed keys exist without verified source ownership"):
        asyncio.run(ingest_paragraph(world.connection, pipeline, request))
    assert _snapshot(world.connection) == before
    world.evidence["unattributed_managed_key_preserved"] = True


def test_pipeline_replay_direct_writer_and_full_paragraph(world, monkeypatch):
    request = world.request(text=("  Long paragraph café.\n" * 1500) + world.run_id)
    writer = AtomicKGWriter(world.connection.connect(), world.settings)
    captured = []
    original = writer.run

    async def record(graph, lexical_graph_config=LexicalGraphConfig()):
        graph = Neo4jGraph.model_validate(graph)
        captured.append(graph.model_copy(deep=True))
        return await original(graph, lexical_graph_config)

    monkeypatch.setattr(writer, "run", record)
    pipeline, llm, embedder = world.pipeline(writer)
    text, metadata = build_plugin_input(request)
    first = asyncio.run(pipeline.run_async(text=text, document_metadata=metadata))
    assert first.result["writer"]["status"] == "SUCCESS"
    assert classify_chunk(world.connection, request).state == COMPLETE
    before = read_inventory(world.connection, request.chunk_id)
    assert len(before["nodes"]) == 5 and len(before["relationships"]) == 6
    chunks = [node for node in before["nodes"] if "Chunk" in node["labels"]]
    assert len(chunks) == 1 and chunks[0]["properties"]["text"] == text
    assert chunks[0]["properties"]["embedding"] == FixedEmbedder.vector
    assert chunks[0]["properties"]["source_sha256"] == request.provenance.source_sha256
    second = asyncio.run(pipeline.run_async(text=text, document_metadata=metadata))
    assert second.result["writer"]["metadata"]["reused"] is True
    assert asyncio.run(original(captured[0])).metadata["reused"] is True
    assert read_inventory(world.connection, request.chunk_id) == before
    assert llm.calls == embedder.calls == 2
    reused = asyncio.run(ingest_paragraph(world.connection, pipeline, request))
    assert reused.reused and llm.calls == embedder.calls == 2
    # Same-name entities across paragraphs remain scoped, with no hidden resolver.
    other = world.request(1)
    assert asyncio.run(ingest_paragraph(world.connection, pipeline, other)).state.state == COMPLETE
    assert len(read_inventory(world.connection, other.chunk_id)["nodes"]) == 5
    assert read_inventory(world.connection, request.chunk_id) == before
    world.evidence.update(inventory=before, replay_unchanged=True, boundary_skipped_provider=True,
                          llm_calls=llm.calls, embedding_calls=embedder.calls)


@pytest.mark.parametrize("stage", ["node", "relationship", "completion"])
def test_transaction_failure_rolls_back_every_stage(world, monkeypatch, stage):
    request = world.request()
    pipeline, llm, embedder = world.pipeline()
    original = TransactionConnection.run_write

    def fail(self, query, parameters=None):
        result = original(self, query, parameters)
        if ((stage == "node" and query.startswith("MERGE (n:"))
            or (stage == "relationship" and "MERGE (a)-[r:" in query)
            or (stage == "completion" and query.startswith("CREATE (m:IngestionCompletion"))):
            raise RuntimeError("injected atomic write failure")
        return result

    monkeypatch.setattr(TransactionConnection, "run_write", fail)
    with pytest.raises(Exception, match="injected atomic write failure"):
        asyncio.run(ingest_paragraph(world.connection, pipeline, request))
    assert classify_chunk(world.connection, request).state == ABSENT
    assert read_inventory(world.connection, request.chunk_id) == {"nodes": [], "relationships": []}
    monkeypatch.setattr(TransactionConnection, "run_write", original)
    assert asyncio.run(ingest_paragraph(world.connection, pipeline, request)).state.state == COMPLETE
    world.evidence.update(rollback_stage=stage, rollback_verified=True)


def test_transient_transaction_retry_does_not_repeat_model_work(world, monkeypatch):
    request = world.request()
    original = TransactionConnection.run_write
    failures = []

    def transient(self, query, parameters=None):
        result = original(self, query, parameters)
        if "MERGE (a)-[r:" in query and not failures:
            failures.append(True)
            raise TransientError("injected retryable write failure")
        return result

    monkeypatch.setattr(TransactionConnection, "run_write", transient)
    pipeline, llm, embedder = world.pipeline()
    assert asyncio.run(ingest_paragraph(world.connection, pipeline, request)).state.state == COMPLETE
    assert failures == [True] and llm.calls == embedder.calls == 1
    world.evidence.update(transient_retry=True, llm_calls=llm.calls, embedding_calls=embedder.calls)


def test_lost_commit_acknowledgment_reconciles_committed_work(world, monkeypatch):
    request = world.request()
    writer = AtomicKGWriter(world.connection.connect(), world.settings)
    original = writer._write
    lost_ack = []

    def write(connection, *args):
        result = original(connection, *args)
        if not lost_ack:
            lost_ack.append(True)
            real_commit = connection.transaction.commit

            def commit():
                real_commit()
                raise ServiceUnavailable("injected lost commit acknowledgment")

            monkeypatch.setattr(connection.transaction, "commit", commit)
        return result

    monkeypatch.setattr(writer, "_write", write)
    pipeline, llm, embedder = world.pipeline(writer)
    result = asyncio.run(ingest_paragraph(world.connection, pipeline, request))
    assert result.state.state == COMPLETE
    assert result.result.result["writer"]["metadata"]["reused"] is True
    assert llm.calls == embedder.calls == 1
    assert len(read_inventory(world.connection, request.chunk_id)["nodes"]) == 5
    world.evidence.update(lost_ack_reconciled=True, provider_calls=llm.calls)


@pytest.mark.parametrize("change", ["text", "source_file", "source_sha256", "record_id", "title"])
def test_changed_input_and_tampered_graph_are_not_replayed(world, change):
    request = world.request()
    pipeline, llm, embedder = world.pipeline()
    asyncio.run(ingest_paragraph(world.connection, pipeline, request))
    before = read_inventory(world.connection, request.chunk_id)
    if change.startswith("source_"):
        changed = request.model_copy(update={"provenance": request.provenance.model_copy(update={change: "changed"})})
    else:
        changed = request.model_copy(update={change: "changed"})
    result = asyncio.run(ingest_paragraph(world.connection, pipeline, changed))
    assert result.state.state == CONFLICT and llm.calls == 1
    text, metadata = build_plugin_input(changed)
    with pytest.raises(Exception, match="conflict"):
        asyncio.run(pipeline.run_async(text=text, document_metadata=metadata))
    assert read_inventory(world.connection, request.chunk_id) == before
    world.connection.run_write(
        "MATCH ()-[r:FOUNDED {ingestion_chunk_id: $id}]->() WITH r LIMIT 1 DELETE r", {"id": request.chunk_id},
    )
    assert classify_chunk(world.connection, request).state == PARTIAL
    after = read_inventory(world.connection, request.chunk_id)
    assert asyncio.run(ingest_paragraph(world.connection, pipeline, request)).state.state == PARTIAL
    assert read_inventory(world.connection, request.chunk_id) == after
    world.evidence.update(changed_field=change, conflict_preserved_graph=True, tampering_detected=True)


@pytest.mark.parametrize("stop", ["before_commit", "after_commit"])
def test_real_process_restart_across_commit_and_checkpoint(world, tmp_path, stop):
    request = world.request()
    request_path, manifest, events = [tmp_path / name for name in ("request.json", "manifest.json", "events.jsonl")]
    request_path.write_text(request.model_dump_json())
    command = [sys.executable, str(Path(__file__).with_name("atomic_ingestion_child.py")),
               str(request_path), str(manifest), str(events)]
    first = subprocess.run(command + ["--stop", stop], capture_output=True, text=True, timeout=45)
    assert first.returncode == (74 if stop == "before_commit" else 73), first.stderr
    checkpoint = json.loads(manifest.read_text())
    assert checkpoint["chunks"][request.chunk_id]["status"] == "in_flight"
    state = classify_chunk(world.connection, request)
    assert state.state == (ABSENT if stop == "before_commit" else COMPLETE)
    inventory = read_inventory(world.connection, request.chunk_id)
    before_events = events.read_text().splitlines()
    second = subprocess.run(command, capture_output=True, text=True, timeout=45)
    assert second.returncode == 0, second.stderr
    report = json.loads(second.stdout)
    assert report["complete_records"] == [request.record_id]
    assert report["outcome"]["interrupted_in_flight"] == [request.chunk_id]
    final = json.loads(manifest.read_text())
    assert final["chunks"][request.chunk_id]["status"] == "verified"
    assert classify_chunk(world.connection, request).state == COMPLETE
    if stop == "after_commit":
        assert report["outcome"]["skipped_verified"] == [request.chunk_id]
        assert events.read_text().splitlines() == before_events
        assert read_inventory(world.connection, request.chunk_id) == inventory
        assert final["chunks"][request.chunk_id]["attempts"] == 1
    else:
        assert len(report["succeeded"]) == 1
        assert final["chunks"][request.chunk_id]["attempts"] == 2
    # A missing manifest and a deliberately fresh manifest both reconcile the
    # committed source, without extraction, embeddings or duplicate graph effects.
    committed = read_inventory(world.connection, request.chunk_id)
    provider_events = events.read_text()
    missing = tmp_path / "missing-manifest.json"
    third = subprocess.run([*command[:3], str(missing), str(events)], capture_output=True, text=True, timeout=45)
    assert third.returncode == 0, third.stderr
    fourth = subprocess.run(command + ["--fresh"], capture_output=True, text=True, timeout=45)
    assert fourth.returncode == 0, fourth.stderr
    assert list(tmp_path.glob("manifest.json.superseded-*"))
    assert events.read_text() == provider_events
    assert read_inventory(world.connection, request.chunk_id) == committed
    world.evidence.update(stop=stop, first_exit=first.returncode, resume=report,
                          interrupted_manifest=checkpoint, final_manifest=final,
                          inventory=committed, missing_manifest=json.loads(third.stdout),
                          fresh_manifest=json.loads(fourth.stdout), provider_events=provider_events.splitlines())


def test_batch_preload_and_fixture_commands_share_atomic_boundary(world, tmp_path):
    dataset = tmp_path / "musique_full_v1.0_dev.jsonl"
    dataset.write_text(json.dumps({"id": world.run_id, "question": "Who founded Aster?",
        "paragraphs": [{"idx": 0, "title": "Aster", "paragraph_text": "Alex founded Aster."}]}) + "\n")
    requests, errors, _ = collect_requests(dataset, 1, 1, "full_dev")
    assert not errors
    world.requests.extend(requests)
    request = requests[0]
    llm, embedder = FixedGraphLLM(world.run_id), FixedEmbedder()
    models = OllamaSettings(host="http://unused", llm_model="fixed", embedding_model="fixed")
    with (
        patch("graphrag_pipeline.extraction.pipeline.OllamaJSONSchemaLLM", return_value=llm),
        patch("graphrag_pipeline.extraction.pipeline.OllamaEmbeddings", return_value=embedder),
        patch("graphrag_pipeline.ingestion.load_musique_dataset.model_settings_from_env", return_value=models),
        patch("graphrag_pipeline.extraction.run_fixture.model_settings_from_env", return_value=models),
    ):
        selected = select_queries([dataset], record_id=world.run_id, preload=True)
        preload = asyncio.run(load_selected_records(selected, neo4j_settings=world.settings,
            openrouter_settings=models, show_progress=False, state_dir=tmp_path / "preload-state"))
        assert preload.succeeded and preload.records_succeeded == 1
        before = read_inventory(world.connection, request.chunk_id)
        batch = asyncio.run(load_dataset(dataset, 1, 1, None, False, show_progress=False,
                                         state_dir=tmp_path / "batch-state"))
        assert batch.succeeded and batch.chunks_skipped_verified == 1
        fixture_path = tmp_path / "fixture.json"
        fixture_path.write_text(request.model_dump_json())
        fixture = asyncio.run(run_fixture(fixture_path))
        assert fixture["reused"] and fixture["status"] == "succeeded"
        assert llm.calls == embedder.calls == 1
        assert read_inventory(world.connection, request.chunk_id) == before
        # Removing a source relationship invalidates completion for every entrypoint.
        world.connection.run_write(
            "MATCH ()-[r:FROM_DOCUMENT {ingestion_chunk_id: $id}]->() DELETE r", {"id": request.chunk_id},
        )
        broken = read_inventory(world.connection, request.chunk_id)
        preload = asyncio.run(load_selected_records(selected, neo4j_settings=world.settings,
            openrouter_settings=models, show_progress=False, state_dir=tmp_path / "preload-state"))
        assert not preload.succeeded and preload.records_succeeded == 0
        assert preload.chunks_needs_investigation
        with pytest.raises(Exception, match="partial"):
            asyncio.run(run_fixture(fixture_path))
        assert llm.calls == embedder.calls == 1
        assert read_inventory(world.connection, request.chunk_id) == broken
    world.evidence.update(preload=preload.to_dict(), batch=batch.to_dict(), fixture=fixture,
                          inventory=before, provider_calls=llm.calls, unresolved_record_incomplete=True)


def test_independent_persisted_graph_audit(world):
    request = world.request()
    pipeline, llm, embedder = world.pipeline()
    asyncio.run(ingest_paragraph(world.connection, pipeline, request))
    assert asyncio.run(ingest_paragraph(world.connection, pipeline, request)).reused
    # Raw graph queries, independent of completion/classifier/inventory helpers.
    sources = world.connection.run_read(
        "MATCH (d:Document {chunk_id: $id}) OPTIONAL MATCH (c:Chunk)-[r:FROM_DOCUMENT]->(d) "
        "RETURN properties(d) AS document, properties(c) AS chunk, elementId(r) AS source_link",
        {"id": request.chunk_id},
    )
    assert len(sources) == 1 and sources[0]["source_link"]
    _, metadata = build_plugin_input(request)
    for props in (sources[0]["document"], sources[0]["chunk"]):
        assert all(props.get(key) == value for key, value in metadata.items())
    assert sources[0]["chunk"]["text"] == request.text
    assert sources[0]["chunk"]["embedding"] == FixedEmbedder.vector
    entities = world.connection.run_read(
        "MATCH (e:__Entity__)-[r:FROM_CHUNK]->(c:Chunk {chunk_id: $id}) "
        "RETURN elementId(e) AS entity, e.name AS name, elementId(r) AS source_link",
        {"id": request.chunk_id},
    )
    assert len(entities) == len({row["entity"] for row in entities}) == 3
    assert len([row for row in entities if row["name"].startswith("Alex ")]) == 2
    relationships = world.connection.run_read(
        "MATCH (a {ingestion_chunk_id: $id})-[r]->(b) "
        "RETURN type(r) AS type, r.ingestion_key AS key, b.ingestion_chunk_id AS target_owner",
        {"id": request.chunk_id},
    )
    assert sorted(row["type"] for row in relationships) == ["FOUNDED", "FOUNDED", "FROM_CHUNK", "FROM_CHUNK", "FROM_CHUNK", "FROM_DOCUMENT"]
    assert len({row["key"] for row in relationships}) == 6
    assert all(row["target_owner"] == request.chunk_id for row in relationships)
    markers = world.connection.run_read(
        "MATCH (m:IngestionCompletion {chunk_id: $id}) RETURN properties(m) AS marker", {"id": request.chunk_id},
    )
    assert len(markers) == 1
    marker = markers[0]["marker"]
    assert json.loads(marker["request_json"]) == request.model_dump(mode="json")
    inventory = json.loads(marker["inventory_json"])
    assert len(inventory["nodes"]) == 5 and len(inventory["relationships"]) == 6
    assert llm.calls == embedder.calls == 1
    world.evidence.update(source=sources, entities=entities, relationships=relationships,
                          completion=marker, provider_calls=llm.calls, independent_audit_passed=True)
