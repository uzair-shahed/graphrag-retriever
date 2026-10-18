"""Versioned paragraph identity and persisted graph inventory.

Completion is evidence of an atomic write, not a substitute for reading the
actual source and graph again. Gold labels and provider settings are not part
of source identity. Existing unmarked data is never adopted automatically.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from neo4j_graphrag.components.text_splitters.base import TextSplitter
from neo4j_graphrag.components.types import TextChunk

from graphrag_pipeline.extraction.schemas import ExtractionRequest

CONTRACT_VERSION = 1
MANAGED_LABEL = "__IngestedNode__"
COMPLETION_LABEL = "IngestionCompletion"
RESERVED = {"ingestion_key", "ingestion_chunk_id"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def request_json(request: ExtractionRequest) -> str:
    return canonical_json(request.model_dump(mode="json"))


def fingerprint(request: ExtractionRequest) -> str:
    return hashlib.sha256(request_json(request).encode("utf-8")).hexdigest()


def managed_key(chunk_id: str, role: str, local_id: str) -> str:
    return hashlib.sha256(canonical_json([chunk_id, role, local_id]).encode("utf-8")).hexdigest()


class ParagraphSplitter(TextSplitter):
    """The validated application paragraph is the extraction unit, unchanged."""

    def iter_chunks(self, text: str):
        if not text:
            raise ValueError("paragraph must not be empty")
        yield TextChunk(text=text, index=0)


def quote_identifier(value: str) -> str:
    if not value or "\x00" in value:
        raise ValueError("empty or NUL-containing graph identifier")
    return "`" + value.replace("`", "``") + "`"


def ensure_schema(connection, relationship_types=()) -> None:
    """Scoped constraints leave legacy labels/data untouched; inspect, don't trust IF NOT EXISTS."""
    definitions = [
        ("ingestion_completion_chunk", "UNIQUENESS", COMPLETION_LABEL, "chunk_id", "NODE"),
        ("ingestion_node_key", "UNIQUENESS", MANAGED_LABEL, "ingestion_key", "NODE"),
    ]
    for rel_type in sorted(set(relationship_types)):
        digest = hashlib.sha256(rel_type.encode()).hexdigest()[:24]
        definitions.append((f"ingestion_rel_{digest}", "RELATIONSHIP_UNIQUENESS", rel_type, "ingestion_key", "RELATIONSHIP"))
    for name, kind, label, prop, entity_type in definitions:
        pattern = f"(n:{quote_identifier(label)})" if entity_type == "NODE" else f"()-[n:{quote_identifier(label)}]-()"
        connection.run_write(
            f"CREATE CONSTRAINT {quote_identifier(name)} IF NOT EXISTS FOR {pattern} REQUIRE n.{prop} IS UNIQUE"
        )
        rows = connection.run_read(
            "SHOW CONSTRAINTS YIELD name, type, entityType, labelsOrTypes, properties "
            "WHERE name = $name RETURN type, entityType, labelsOrTypes, properties", {"name": name},
        )
        # Neo4j 2026 reports *_PROPERTY_UNIQUENESS; older Neo4j APIs
        # use UNIQUENESS / RELATIONSHIP_UNIQUENESS for the same constraints.
        for row in rows:
            row["type"] = {"NODE_PROPERTY_UNIQUENESS": "UNIQUENESS",
                           "RELATIONSHIP_PROPERTY_UNIQUENESS": "RELATIONSHIP_UNIQUENESS"}.get(row["type"], row["type"])
        expected = {"type": kind, "entityType": entity_type, "labelsOrTypes": [label], "properties": [prop]}
        if rows != [expected]:
            raise RuntimeError(f"incompatible ingestion constraint {name}: expected {expected}, got {rows}")


def read_inventory(connection, chunk_id: str) -> dict[str, Any]:
    params = {"chunk_id": chunk_id}
    nodes = connection.run_read(
        "MATCH (n) WHERE n.ingestion_chunk_id = $chunk_id "
        "RETURN n.ingestion_key AS key, labels(n) AS labels, properties(n) AS properties", params,
    )
    for node in nodes:
        node["labels"].sort()
    relationships = connection.run_read(
        "MATCH (a)-[r]->(b) WHERE a.ingestion_chunk_id = $chunk_id "
        "OR b.ingestion_chunk_id = $chunk_id OR r.ingestion_chunk_id = $chunk_id "
        "RETURN r.ingestion_key AS key, a.ingestion_key AS source, b.ingestion_key AS target, "
        "type(r) AS type, properties(r) AS properties", params,
    )
    return {
        "nodes": sorted(nodes, key=canonical_json),
        "relationships": sorted(relationships, key=canonical_json),
    }


def inventory_json(connection, chunk_id: str) -> str:
    return canonical_json(read_inventory(connection, chunk_id))


def completion_rows(connection, chunk_id: str) -> list[dict[str, Any]]:
    return connection.run_read(
        "MATCH (m:IngestionCompletion {chunk_id: $chunk_id}) RETURN properties(m) AS completion",
        {"chunk_id": chunk_id},
    )
