"""Resumable ingestion tracking, completion markers, and uniqueness constraints.

Provides idempotency and crash recovery for MuSiQue paragraph ingestion:
1. Enforces uniqueness constraint on (:Document {paragraph_hash}).
2. Records completion markers (ingest_status='complete', pipeline_version, schema_hash).
3. Queries completed paragraphs to skip already-extracted text ($0 rerun cost).
4. Cleans up uncommitted / partial paragraph writes before retrying.
"""

from __future__ import annotations

import logging
from typing import Any

from neo4j import Driver

from graphrag_pipeline.extraction.schema_v1 import SCHEMA_V1_HASH

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "1.0.0"
SCHEMA_HASH = SCHEMA_V1_HASH


def ensure_paragraph_hash_constraint(driver: Driver) -> None:
    """Ensure the unique constraint on Document.paragraph_hash exists."""
    query = (
        "CREATE CONSTRAINT paragraph_hash_unique IF NOT EXISTS "
        "FOR (d:Document) REQUIRE d.paragraph_hash IS UNIQUE"
    )
    with driver.session() as session:
        session.run(query).consume()


def load_completed_paragraph_hashes(driver: Driver) -> set[str]:
    """Query Neo4j for paragraph hashes marked as complete."""
    query = (
        "MATCH (d:Document {ingest_status: 'complete'}) "
        "RETURN d.paragraph_hash AS hash"
    )
    with driver.session() as session:
        result = session.run(query)
        return {record["hash"] for record in result if record.get("hash")}


def mark_paragraph_complete(
    driver: Driver,
    paragraph_hash: str,
    pipeline_version: str = PIPELINE_VERSION,
    schema_hash: str = SCHEMA_HASH,
) -> None:
    """Mark a Document node as completely ingested after writer verification."""
    query = (
        "MATCH (d:Document {paragraph_hash: $h}) "
        "SET d.ingest_status = 'complete', "
        "    d.pipeline_version = $v, "
        "    d.schema_hash = $s"
    )
    with driver.session() as session:
        session.run(
            query,
            {"h": paragraph_hash, "v": pipeline_version, "s": schema_hash},
        ).consume()


def cleanup_incomplete_paragraph(driver: Driver, paragraph_hash: str) -> int:
    """Delete any partially written Document and associated Chunk/Entities for paragraph_hash."""
    query = (
        "MATCH (d:Document {paragraph_hash: $h}) "
        "WHERE d.ingest_status IS NULL OR d.ingest_status <> 'complete' "
        "OPTIONAL MATCH (c:Chunk)-[:FROM_DOCUMENT]->(d) "
        "OPTIONAL MATCH (e)-[:FROM_CHUNK]->(c) "
        "DETACH DELETE d, c, e "
        "RETURN count(d) AS deleted_count"
    )
    with driver.session() as session:
        result = session.run(query, {"h": paragraph_hash})
        record = result.single()
        return record["deleted_count"] if record else 0


def cleanup_all_incomplete_paragraphs(driver: Driver) -> int:
    """Clean up all incomplete Document nodes without complete marker."""
    query = (
        "MATCH (d:Document) "
        "WHERE d.ingest_status IS NULL OR d.ingest_status <> 'complete' "
        "OPTIONAL MATCH (c:Chunk)-[:FROM_DOCUMENT]->(d) "
        "OPTIONAL MATCH (e)-[:FROM_CHUNK]->(c) "
        "DETACH DELETE d, c, e "
        "RETURN count(d) AS deleted_count"
    )
    with driver.session() as session:
        result = session.run(query)
        record = result.single()
        return record["deleted_count"] if record else 0
