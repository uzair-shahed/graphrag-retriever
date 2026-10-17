"""Integration tests for seed-chunk entity-subgraph traversal against real Neo4j.

Builds a small synthetic fixture using the real ``Chunk``/``Document``/
``__Entity__`` labels and ``FROM_CHUNK``/``FROM_DOCUMENT`` relationship types
(the traversal Cypher in ``graphrag_pipeline.retrieval`` hardcodes these), then
runs the production ``_retrieval_query_vector_cypher`` projection directly
against a chosen seed chunk -- bypassing embedding-based vector search, which
is covered separately by the live retrieval harness. Skips automatically when
Neo4j is unreachable or unconfigured, matching
``test_neo4j_connection_smoke.py``. All fixture data is created under a unique
``fixture_run_id`` and deleted in a ``finally`` block.
"""

from __future__ import annotations

import uuid

import pytest
from neo4j.exceptions import ServiceUnavailable

from graphrag_pipeline.config import Neo4jSettings
from graphrag_pipeline.retrieval import _retrieval_query_vector_cypher
from graphrag_pipeline.storage.neo4j_connection import Neo4jConnection


def _load_connection_or_skip() -> Neo4jConnection:
    try:
        settings = Neo4jSettings.from_env()
    except ValueError:
        pytest.skip("NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD not configured; skipping integration test.")
    connection = Neo4jConnection(settings)
    try:
        connection.verify_connectivity()
    except ServiceUnavailable:
        pytest.skip("Local Neo4j is not reachable; skipping integration test.")
    return connection


def _build_fixture(connection: Neo4jConnection, run_id: str) -> None:
    """Create two paragraphs' worth of chunks/documents/entities for one test run.

    Record A (seed under test): a cycle (e1->e2->e3->e1), a second path
    (e1->e4), and a chain six hops deep beyond the seed's own entities
    (e4->e5->...->e9) with no hop-depth ceiling expected to stop it.

    Record B (distractor): its own entity, plus ``e2`` is deliberately given a
    *second* ``FROM_CHUNK`` edge into record B's chunk to simulate a future
    shared/resolved entity -- traversal from record A's seed must not reach
    record B's distractor entity through it.
    """
    connection.run_write(
        """
        MERGE (docA:Document {chunk_id: $chunk_a, run_id: $run_id})
        SET docA.record_id = $record_a, docA.document_id = $chunk_a, docA.title = 'Fixture A'
        MERGE (chunkA:Chunk {run_id: $run_id, seed: 'a'})
        SET chunkA.text = 'Fixture paragraph A'
        MERGE (chunkA)-[:FROM_DOCUMENT]->(docA)

        MERGE (docB:Document {chunk_id: $chunk_b, run_id: $run_id})
        SET docB.record_id = $record_b, docB.document_id = $chunk_b, docB.title = 'Fixture B'
        MERGE (chunkB:Chunk {run_id: $run_id, seed: 'b'})
        SET chunkB.text = 'Fixture paragraph B'
        MERGE (chunkB)-[:FROM_DOCUMENT]->(docB)

        MERGE (e1:__Entity__:FixtureEntity {run_id: $run_id, name: 'e1'})
        MERGE (e2:__Entity__:FixtureEntity {run_id: $run_id, name: 'e2'})
        MERGE (e3:__Entity__:FixtureEntity {run_id: $run_id, name: 'e3'})
        MERGE (e4:__Entity__:FixtureEntity {run_id: $run_id, name: 'e4'})
        MERGE (e5:__Entity__:FixtureEntity {run_id: $run_id, name: 'e5'})
        MERGE (e6:__Entity__:FixtureEntity {run_id: $run_id, name: 'e6'})
        MERGE (e7:__Entity__:FixtureEntity {run_id: $run_id, name: 'e7'})
        MERGE (e8:__Entity__:FixtureEntity {run_id: $run_id, name: 'e8'})
        MERGE (e9:__Entity__:FixtureEntity {run_id: $run_id, name: 'e9'})
        MERGE (ex:__Entity__:FixtureEntity {run_id: $run_id, name: 'ex_distractor'})

        MERGE (e1)-[:FROM_CHUNK]->(chunkA)
        MERGE (e2)-[:FROM_CHUNK]->(chunkA)
        MERGE (e3)-[:FROM_CHUNK]->(chunkA)
        MERGE (e4)-[:FROM_CHUNK]->(chunkA)
        MERGE (e2)-[:FROM_CHUNK]->(chunkB)
        MERGE (ex)-[:FROM_CHUNK]->(chunkB)

        MERGE (e1)-[:RELATED_TO]->(e2)
        MERGE (e2)-[:RELATED_TO]->(e3)
        MERGE (e3)-[:RELATED_TO]->(e1)
        MERGE (e1)-[:ALSO_RELATED_TO]->(e4)
        MERGE (e4)-[:CHAINS_TO]->(e5)
        MERGE (e5)-[:CHAINS_TO]->(e6)
        MERGE (e6)-[:CHAINS_TO]->(e7)
        MERGE (e7)-[:CHAINS_TO]->(e8)
        MERGE (e8)-[:CHAINS_TO]->(e9)
        """,
        {
            "run_id": run_id,
            "chunk_a": f"fixture:{run_id}:a",
            "chunk_b": f"fixture:{run_id}:b",
            "record_a": f"fixture-record-{run_id}-a",
            "record_b": f"fixture-record-{run_id}-b",
        },
    )


def _delete_fixture(connection: Neo4jConnection, run_id: str) -> None:
    connection.run_write(
        "MATCH (n) WHERE n.run_id = $run_id DETACH DELETE n",
        {"run_id": run_id},
    )


def _traverse_seed(connection: Neo4jConnection, run_id: str, seed: str, node_limit: int) -> dict:
    relationship_filter = "FROM_CHUNK<|RELATED_TO|ALSO_RELATED_TO|CHAINS_TO"
    query = (
        "MATCH (node:Chunk {run_id: $run_id, seed: $seed}) "
        "WITH node, 1.0 AS score " + _retrieval_query_vector_cypher(relationship_filter)
    )
    rows = connection.run_read(query, {"run_id": run_id, "seed": seed, "node_limit": node_limit, "query_limit": node_limit + 1})
    assert len(rows) == 1
    return rows[0]


def test_traversal_returns_full_component_with_cycle_and_deep_chain() -> None:
    """A cycle and a six-hop chain are both returned, deduplicated, with no hop cap."""
    connection = _load_connection_or_skip()
    run_id = f"task1-{uuid.uuid4().hex}"
    try:
        _build_fixture(connection, run_id)
        row = _traverse_seed(connection, run_id, "a", node_limit=100)

        names = {node["name"] for node in row["graph_nodes"]}
        assert names == {"e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9"}
        assert row["graph_complete"] is True

        rel_types_by_pair = {(r["type"]) for r in row["graph_relationships"]}
        assert "RELATED_TO" in rel_types_by_pair
        assert "CHAINS_TO" in rel_types_by_pair
        # the cycle e1->e2->e3->e1 must not duplicate the RELATED_TO edges
        related_to_edges = [r for r in row["graph_relationships"] if r["type"] == "RELATED_TO"]
        assert len(related_to_edges) == 3
    finally:
        _delete_fixture(connection, run_id)
        connection.close()


def test_traversal_excludes_distractor_with_no_domain_relationship() -> None:
    """An entity connected only via another record's FROM_CHUNK edge, with no domain relationship, is never returned."""
    connection = _load_connection_or_skip()
    run_id = f"task1-{uuid.uuid4().hex}"
    try:
        _build_fixture(connection, run_id)
        row = _traverse_seed(connection, run_id, "a", node_limit=100)

        names = {node["name"] for node in row["graph_nodes"]}
        assert "ex_distractor" not in names
    finally:
        _delete_fixture(connection, run_id)
        connection.close()


def test_traversal_merges_shared_entity_component_but_keeps_seed_specific_provenance() -> None:
    """A shared entity legitimately merges its connected domain neighbors, but chunk/document identity stays seed-specific.

    ``e2`` is connected to record A's entities by real domain relationships
    *and* carries its own ``FROM_CHUNK`` edge into record B -- simulating a
    future resolved/shared entity. Seeding from record B's chunk correctly
    reaches the merged component through that genuine connectivity (this is
    not a boundary violation: the edges are real), and also correctly
    includes ``ex_distractor`` (record B's own directly-extracted entity).
    The safeguard that matters is that the returned chunk *text and
    provenance* (``chunk_id``/``document_id``) always belong to the seed
    chunk, never to the other record -- proven separately by the exclusion
    test above, which seeds record A and never sees ``ex_distractor`` because
    it has no domain-relationship path from record A's entities.
    """
    connection = _load_connection_or_skip()
    run_id = f"task1-{uuid.uuid4().hex}"
    try:
        _build_fixture(connection, run_id)
        row = _traverse_seed(connection, run_id, "b", node_limit=100)

        names = {node["name"] for node in row["graph_nodes"]}
        assert {"e1", "e2", "e3", "e9", "ex_distractor"} <= names
        assert row["chunk_id"] == f"fixture:{run_id}:b"
    finally:
        _delete_fixture(connection, run_id)
        connection.close()


def test_traversal_reports_incomplete_when_resource_budget_is_exceeded() -> None:
    """A node budget smaller than the true component is reported as incomplete, not silently trimmed-complete."""
    connection = _load_connection_or_skip()
    run_id = f"task1-{uuid.uuid4().hex}"
    try:
        _build_fixture(connection, run_id)
        row = _traverse_seed(connection, run_id, "a", node_limit=3)

        assert row["graph_complete"] is False
        assert len(row["graph_nodes"]) <= 3
    finally:
        _delete_fixture(connection, run_id)
        connection.close()
