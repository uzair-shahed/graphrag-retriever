"""Atomic source-scoped writer for SimpleKGPipeline (entity resolution disabled)."""

from __future__ import annotations

import math

from neo4j import Driver
from pydantic import validate_call
from neo4j_graphrag.components.kg_writer import KGWriter, KGWriterModel
from neo4j_graphrag.components.types import LexicalGraphConfig, Neo4jGraph

from graphrag_pipeline.config import Neo4jSettings
from graphrag_pipeline.extraction.idempotency import (
    CONTRACT_VERSION, MANAGED_LABEL, RESERVED, canonical_json, ensure_schema,
    fingerprint, inventory_json, managed_key, quote_identifier, request_json,
)
from graphrag_pipeline.extraction.plugin_input import build_plugin_input
from graphrag_pipeline.extraction.schemas import ExtractionRequest, Provenance
from graphrag_pipeline.ingestion.retry_policy import PipelineStageError
from graphrag_pipeline.ingestion.reconcile import ABSENT, COMPLETE, classify_chunk
from graphrag_pipeline.storage.neo4j_connection import Neo4jConnection


class IngestionStateError(RuntimeError):
    """Persisted source identity or graph state prevents safe ingestion."""


class TransactionConnection:
    """Use the same explicit transaction for reconciliation and every graph write."""

    def __init__(self, transaction):
        self.transaction = transaction

    def run_read(self, query, parameters=None):
        return [record.data() for record in self.transaction.run(query, parameters or {})]

    run_write = run_read


def prepare_graph(graph: Neo4jGraph, config: LexicalGraphConfig):
    """Validate source structure and construct immutable transaction input."""
    if config != LexicalGraphConfig():
        raise ValueError("atomic ingestion requires the standard lexical graph labels and relationships")
    documents = [node for node in graph.nodes if node.label == "Document"]
    chunks = [node for node in graph.nodes if node.label == "Chunk"]
    if len(documents) != 1 or len(chunks) != 1:
        raise ValueError("expected exactly one Document and one full-paragraph Chunk")
    document, chunk = documents[0], chunks[0]
    metadata = document.properties
    request = ExtractionRequest(
        chunk_id=metadata["chunk_id"], record_id=metadata["record_id"],
        document_id=metadata["document_id"], paragraph_idx=int(metadata["paragraph_idx"]),
        title=metadata.get("title"), text=chunk.properties["text"],
        provenance=Provenance(
            source_dataset=metadata["source_dataset"], source_split=metadata["source_split"],
            source_file=metadata["source_file"], source_sha256=metadata["source_sha256"],
            source_line=int(metadata["source_line"]),
        ),
    )
    _, provenance = build_plugin_input(request)
    node_ids = [node.id for node in graph.nodes]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("duplicate plugin node ID")
    keys = {node.id: managed_key(request.chunk_id, "entity", node.id) for node in graph.nodes}
    keys[document.id] = managed_key(request.chunk_id, "document", "source")
    keys[chunk.id] = managed_key(request.chunk_id, "chunk", "source")
    nodes = []
    for node in graph.nodes:
        if not node.id or node.label in (MANAGED_LABEL, "IngestionCompletion", "__KGBuilder__", "__Entity__"):
            raise ValueError("reserved label or empty node ID in extraction output")
        quote_identifier(node.label)
        if RESERVED & (node.properties.keys() | node.embedding_properties.keys()):
            raise ValueError("provider output uses reserved ingestion properties")
        properties = dict(node.properties)
        if node in (document, chunk):
            properties.update(provenance)
        if node.properties.keys() & node.embedding_properties.keys():
            raise ValueError("embedding properties must not overwrite ordinary node properties")
        properties.update(node.embedding_properties)
        for embedding in node.embedding_properties.values():
            if not embedding or any(isinstance(v, bool) or not math.isfinite(v) for v in embedding):
                raise ValueError("invalid embedding")
        properties.update(ingestion_key=keys[node.id], ingestion_chunk_id=request.chunk_id)
        properties = {key: value for key, value in properties.items() if value is not None}
        labels = [node.label, MANAGED_LABEL, "__KGBuilder__"]
        if node not in (document, chunk):
            labels.append("__Entity__")
        nodes.append({"key": keys[node.id], "labels": labels, "properties": properties})
    if not chunk.embedding_properties.get("embedding"):
        raise ValueError("paragraph Chunk requires an embedding")
    relationships = []
    seen = set()
    source_links = []
    entity_links = []
    for rel in graph.relationships:
        if rel.start_node_id not in keys or rel.end_node_id not in keys:
            raise ValueError("relationship endpoint missing from graph")
        if RESERVED & (rel.properties.keys() | rel.embedding_properties.keys()):
            raise ValueError("relationship uses reserved ingestion properties")
        quote_identifier(rel.type)
        identity = (keys[rel.start_node_id], rel.type, keys[rel.end_node_id])
        if identity in seen:
            raise ValueError("duplicate relationship in extraction output")
        seen.add(identity)
        if rel.type == "FROM_DOCUMENT":
            source_links.append((rel.start_node_id, rel.end_node_id))
        elif rel.type == "FROM_CHUNK":
            entity_links.append((rel.start_node_id, rel.end_node_id))
        elif rel.start_node_id in (document.id, chunk.id) or rel.end_node_id in (document.id, chunk.id):
            raise ValueError("domain relationship cannot target lexical source nodes")
        if rel.properties.keys() & rel.embedding_properties.keys():
            raise ValueError("embedding properties must not overwrite ordinary relationship properties")
        properties = {key: value for key, value in {**rel.properties, **rel.embedding_properties}.items() if value is not None}
        for embedding in rel.embedding_properties.values():
            if not embedding or any(not math.isfinite(v) for v in embedding):
                raise ValueError("invalid relationship embedding")
        key = managed_key(request.chunk_id, "relationship", canonical_json(identity))
        properties.update(ingestion_key=key, ingestion_chunk_id=request.chunk_id)
        relationships.append({"key": key, "source": identity[0], "target": identity[2],
                              "type": rel.type, "properties": properties})
    if source_links != [(chunk.id, document.id)]:
        raise ValueError("expected exactly one Chunk-to-Document source link")
    entities = set(node_ids) - {document.id, chunk.id}
    if set(entity_links) != {(entity, chunk.id) for entity in entities} or len(entity_links) != len(entities):
        raise ValueError("every extracted entity must link to its source Chunk exactly once")
    # Reject non-finite properties before any database call as well.
    canonical_json([nodes, relationships])
    return request, nodes, relationships


class AtomicKGWriter(KGWriter):
    def __init__(self, driver: Driver, settings: Neo4jSettings | None = None):
        # Credentials are unused when borrowing the caller's driver.
        settings = settings or Neo4jSettings("", "", "", "neo4j", 5.0, 15.0, 10.0)
        self.connection = Neo4jConnection(settings, driver=driver)

    @validate_call
    async def run(self, graph: Neo4jGraph, lexical_graph_config: LexicalGraphConfig = LexicalGraphConfig()) -> KGWriterModel:
        try:
            request, nodes, relationships = prepare_graph(graph, lexical_graph_config)
        except (ValueError, KeyError) as error:
            raise PipelineStageError("graph_validation", str(error)) from error
        try:
            ensure_schema(self.connection, [rel["type"] for rel in relationships])
            return self.connection.run_transaction(lambda tx: self._write(TransactionConnection(tx), request, nodes, relationships))
        except Exception as error:
            raise PipelineStageError("write", str(error)) from error

    def _write(self, connection, request, nodes, relationships) -> KGWriterModel:
        state = classify_chunk(connection, request)
        if state.state == COMPLETE:
            return KGWriterModel(status="SUCCESS", metadata={"reused": True, "chunk_id": request.chunk_id})
        if state.state != ABSENT:
            raise IngestionStateError(f"{state.state}: {state.reason}")
        existing = connection.run_read(
            f"MATCH (n:{MANAGED_LABEL}) WHERE n.ingestion_key IN $keys RETURN count(n) AS count",
            {"keys": [node["key"] for node in nodes]},
        )
        if existing != [{"count": 0}]:
            raise IngestionStateError("partial: managed keys exist without verified source ownership; audit required")
        for node in nodes:
            labels = ":".join(quote_identifier(label) for label in node["labels"])
            rows = connection.run_write(
                f"MERGE (n:{MANAGED_LABEL} {{ingestion_key: $key}}) "
                f"SET n:{labels} SET n += $properties RETURN n.ingestion_key AS key", node,
            )
            if len(rows) != 1:
                raise IngestionStateError("node write did not persist exactly one node")
        for rel in relationships:
            rows = connection.run_write(
                f"MATCH (a:{MANAGED_LABEL} {{ingestion_key: $source}}), (b:{MANAGED_LABEL} {{ingestion_key: $target}}) "
                f"MERGE (a)-[r:{quote_identifier(rel['type'])} {{ingestion_key: $key}}]->(b) "
                "SET r += $properties RETURN r.ingestion_key AS key", rel,
            )
            if len(rows) != 1:
                raise IngestionStateError("relationship write did not persist exactly one relationship")
        # read_inventory sorts labels, so normalize the prepared inventory too.
        expected_nodes = [{**node, "labels": sorted(node["labels"])} for node in nodes]
        expected = canonical_json({"nodes": sorted(expected_nodes, key=canonical_json),
                                   "relationships": sorted(relationships, key=canonical_json)})
        if inventory_json(connection, request.chunk_id) != expected:
            raise IngestionStateError("written graph differs from prepared graph inventory")
        connection.run_write(
            "CREATE (m:IngestionCompletion {chunk_id: $chunk_id, fingerprint: $fingerprint, "
            "request_json: $request_json, contract_version: $version, inventory_json: $inventory})",
            {"chunk_id": request.chunk_id, "fingerprint": fingerprint(request), "request_json": request_json(request),
             "version": CONTRACT_VERSION, "inventory": expected},
        )
        state = classify_chunk(connection, request)
        if state.state != COMPLETE:
            raise IngestionStateError(f"atomic verification failed: {state.state}: {state.reason}")
        return KGWriterModel(status="SUCCESS", metadata={"reused": False, "chunk_id": request.chunk_id})
