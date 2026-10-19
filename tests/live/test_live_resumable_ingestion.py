"""Live integration tests for resumable ingestion, completion markers, and uniqueness backstop."""

import os
from pathlib import Path

import pytest
from neo4j.exceptions import ClientError, ServiceUnavailable

from graphrag_pipeline.config import Neo4jSettings
from graphrag_pipeline.extraction.schemas import ExtractionRequest, Provenance
from graphrag_pipeline.ingestion.record_adapter import compute_paragraph_hash
from graphrag_pipeline.storage.neo4j_connection import Neo4jConnection
from graphrag_pipeline.storage.resumable_ingestion import (
    cleanup_all_incomplete_paragraphs,
    cleanup_incomplete_paragraph,
    ensure_paragraph_hash_constraint,
    load_completed_paragraph_hashes,
    mark_paragraph_complete,
)


def _get_neo4j_driver():
    os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
    os.environ.setdefault("NEO4J_USERNAME", "neo4j")
    os.environ.setdefault("NEO4J_PASSWORD", "changeme")
    os.environ.setdefault("NEO4J_DATABASE", "neo4j")

    try:
        settings = Neo4jSettings.from_env()
    except ValueError as e:
        pytest.skip(f"Neo4j not configured: {e}")

    conn = Neo4jConnection(settings)
    try:
        conn.verify_connectivity()
    except ServiceUnavailable:
        pytest.skip("Neo4j not reachable")
    return conn


@pytest.mark.live
def test_uniqueness_backstop_and_resumable_markers():
    """Verify unique constraint rejects duplicate Document hashes and marker enables skipping."""
    conn = _get_neo4j_driver()
    driver = conn.connect()

    try:
        # 1. Ensure constraint exists
        ensure_paragraph_hash_constraint(driver)

        # 2. Clear any test documents with test hash
        test_hash = compute_paragraph_hash("Live Test Title", "Live Test Paragraph Text")
        with driver.session() as s:
            s.run("MATCH (d:Document {paragraph_hash: $h}) DETACH DELETE d", {"h": test_hash})

        # 3. Create initial Document
        with driver.session() as s:
            s.run("CREATE (d:Document {paragraph_hash: $h, title: 'Live Test Title'})", {"h": test_hash})

        # Count nodes with this hash
        with driver.session() as s:
            cnt_1 = s.run("MATCH (d:Document {paragraph_hash: $h}) RETURN count(d) AS c", {"h": test_hash}).single()["c"]
        assert cnt_1 == 1

        # 4. Forced duplicate MUST raise ClientError (ConstraintValidationFailed)
        with pytest.raises(ClientError):
            with driver.session() as s:
                s.run("CREATE (d:Document {paragraph_hash: $h, title: 'Duplicate Title'})", {"h": test_hash})

        # Count must remain 1 (uniqueness backstop refuses duplicate)
        with driver.session() as s:
            cnt_after = s.run("MATCH (d:Document {paragraph_hash: $h}) RETURN count(d) AS c", {"h": test_hash}).single()["c"]
        assert cnt_after == 1

        # 5. Check incomplete cleanup: Document without complete marker is cleaned up
        cleaned = cleanup_incomplete_paragraph(driver, test_hash)
        assert cleaned == 1
        with driver.session() as s:
            cnt_clean = s.run("MATCH (d:Document {paragraph_hash: $h}) RETURN count(d) AS c", {"h": test_hash}).single()["c"]
        assert cnt_clean == 0

        # 6. Recreate Document and set completion marker
        with driver.session() as s:
            s.run("CREATE (d:Document {paragraph_hash: $h, title: 'Live Test Title'})", {"h": test_hash})
        mark_paragraph_complete(driver, test_hash)

        # 7. Completed hashes query loads the marked hash
        completed = load_completed_paragraph_hashes(driver)
        assert test_hash in completed

        # 8. Cleanup does NOT touch completed document
        cleaned_complete = cleanup_incomplete_paragraph(driver, test_hash)
        assert cleaned_complete == 0
        with driver.session() as s:
            cnt_still_there = s.run("MATCH (d:Document {paragraph_hash: $h}) RETURN count(d) AS c", {"h": test_hash}).single()["c"]
        assert cnt_still_there == 1

        # Clean up test node
        with driver.session() as s:
            s.run("MATCH (d:Document {paragraph_hash: $h}) DETACH DELETE d", {"h": test_hash})

    finally:
        conn.close()
