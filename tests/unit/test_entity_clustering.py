"""Unit tests for reversible exact-name entity clustering (Step M6.2)."""

from __future__ import annotations

import uuid
import pytest

from graphrag_pipeline.config import Neo4jSettings
from graphrag_pipeline.linking.clustering import (
    create_exact_name_clusters,
    exact_name_clustering,
    remove_entity_clusters,
)
from graphrag_pipeline.storage.neo4j_connection import Neo4jConnection


def test_reversible_exact_name_clustering_restores_graph_exactly() -> None:
    """Verify that creating exact-name clusters and undoing restores graph node/rel counts exactly."""
    try:
        settings = Neo4jSettings.from_env()
    except ValueError:
        pytest.skip("Local Neo4j instance not configured.")

    with Neo4jConnection(settings) as conn:
        driver = conn.connect()

        # Count nodes and relationships before
        nodes_before = driver.execute_query("MATCH (n) RETURN count(n) AS c")[0][0]["c"]
        rels_before = driver.execute_query("MATCH ()-[r]->() RETURN count(r) AS c")[0][0]["c"]

        run_id = f"test_run_{uuid.uuid4().hex[:8]}"
        summary = create_exact_name_clusters(driver, run_id=run_id)

        assert summary.run_id == run_id
        if summary.clusters_created > 0:
            assert summary.members_linked >= 2 * summary.clusters_created
            assert summary.largest_cluster_size >= 2

            # Nodes and rels increased during clustering
            nodes_during = driver.execute_query("MATCH (n) RETURN count(n) AS c")[0][0]["c"]
            rels_during = driver.execute_query("MATCH ()-[r]->() RETURN count(r) AS c")[0][0]["c"]
            assert nodes_during == nodes_before + summary.clusters_created
            assert rels_during == rels_before + summary.members_linked

        # Undo clustering
        deleted = remove_entity_clusters(driver, run_id=run_id)
        assert deleted == summary.clusters_created

        # Verify exact counts after undo
        nodes_after = driver.execute_query("MATCH (n) RETURN count(n) AS c")[0][0]["c"]
        rels_after = driver.execute_query("MATCH ()-[r]->() RETURN count(r) AS c")[0][0]["c"]

        assert nodes_after == nodes_before, f"Node count mismatch: {nodes_before} != {nodes_after}"
        assert rels_after == rels_before, f"Rel count mismatch: {rels_before} != {rels_after}"


def test_exact_name_clustering_context_manager() -> None:
    """Verify exact_name_clustering context manager automatically tears down on exit."""
    try:
        settings = Neo4jSettings.from_env()
    except ValueError:
        pytest.skip("Local Neo4j instance not configured.")

    with Neo4jConnection(settings) as conn:
        driver = conn.connect()
        run_id = f"test_cm_{uuid.uuid4().hex[:8]}"

        with exact_name_clustering(driver, run_id=run_id) as summary:
            assert summary.run_id == run_id
            active = driver.execute_query(
                "MATCH (k:EntityCluster {run_id: $run}) RETURN count(k) AS c",
                run=run_id,
            )[0][0]["c"]
            assert active == summary.clusters_created

        # Outside block, all clusters for this run_id are deleted
        remaining = driver.execute_query(
            "MATCH (k:EntityCluster {run_id: $run}) RETURN count(k) AS c",
            run=run_id,
        )[0][0]["c"]
        assert remaining == 0


def test_homonyms_with_different_types_never_linked() -> None:
    """Verify that entities with the same name but different types are never linked into the same cluster."""
    try:
        settings = Neo4jSettings.from_env()
    except ValueError:
        pytest.skip("Local Neo4j instance not configured.")

    with Neo4jConnection(settings) as conn:
        driver = conn.connect()
        test_run = f"test_homonym_{uuid.uuid4().hex[:8]}"

        # Create 2 Location entities named 'Paris' and 2 Person entities named 'Paris'
        setup_query = """
        CREATE (l1:__Entity__:Location {name: 'Paris', test_id: $run})
        CREATE (l2:__Entity__:Location {name: 'Paris', test_id: $run})
        CREATE (p1:__Entity__:Person {name: 'Paris', test_id: $run})
        CREATE (p2:__Entity__:Person {name: 'Paris', test_id: $run})
        """
        driver.execute_query(setup_query, run=test_run)

        try:
            with exact_name_clustering(driver, run_id=test_run) as summary:
                # Find clusters for 'paris'
                clusters = driver.execute_query(
                    "MATCH (k:EntityCluster {run_id: $run, key: 'paris'}) "
                    "RETURN k.type AS type, k.size AS size",
                    run=test_run,
                )[0]

                assert len(clusters) == 2, f"Expected 2 separate clusters for 'paris', got {len(clusters)}"
                cluster_types = {c["type"] for c in clusters}
                assert cluster_types == {"Location", "Person"}

                # Assert no SAME_AS edge crosses between Location and Person
                cross_links = driver.execute_query(
                    "MATCH (l:Location)-[:SAME_AS]->(k:EntityCluster)<-[:SAME_AS]-(p:Person) "
                    "WHERE k.run_id = $run "
                    "RETURN count(*) AS count",
                    run=test_run,
                )[0][0]["count"]
                assert cross_links == 0, "Location and Person were erroneously linked together!"
        finally:
            driver.execute_query("MATCH (e {test_id: $run}) DETACH DELETE e", run=test_run)
