"""Reversible exact-name entity clustering (Step M6.2).

Groups entities sharing the same domain type and normalized name into
EntityCluster nodes with SAME_AS relationships. Scoped by run_id so
clustering can be undone cleanly and completely.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from neo4j import Driver

logger = logging.getLogger(__name__)


@dataclass
class ClusterSummary:
    """Summary of created entity clusters and SAME_AS links."""

    run_id: str
    clusters_created: int
    members_linked: int
    largest_cluster_size: int
    types_clustered: dict[str, int] = field(default_factory=dict)


def create_exact_name_clusters(
    driver: Driver,
    run_id: str | None = None,
    database: str = "neo4j",
) -> ClusterSummary:
    """Create reversible EntityCluster nodes and SAME_AS edges for identical (type, normalized_name).

    Args:
        driver: Active Neo4j driver.
        run_id: Identifier scoping this clustering pass. Generated automatically if omitted.
        database: Target Neo4j database name.

    Returns:
        ClusterSummary with creation metrics.
    """
    resolved_run = run_id or f"run_{uuid.uuid4().hex[:12]}"

    create_query = """
    MATCH (e:__Entity__) WHERE e.name IS NOT NULL
    WITH [l IN labels(e) WHERE NOT l STARTS WITH '__'][0] AS type,
         toLower(trim(e.name)) AS key, collect(e) AS members
    WHERE size(members) > 1 AND type IS NOT NULL
    CREATE (k:EntityCluster {
        run_id: $run,
        method: 'exact',
        type: type,
        key: key,
        size: size(members)
    })
    WITH k, members, type
    UNWIND members AS m
    CREATE (m)-[:SAME_AS {
        run_id: $run,
        decision: 'link',
        score: 1.0,
        reason: 'same type, same normalized name'
    }]->(k)
    RETURN count(distinct k) AS clusters, count(m) AS links, max(size(members)) AS max_size
    """

    res, _, _ = driver.execute_query(create_query, run=resolved_run, database_=database)

    clusters = res[0]["clusters"] if res else 0
    links = res[0]["links"] if res else 0
    max_size = res[0]["max_size"] or 0

    # Type breakdown
    type_query = """
    MATCH (k:EntityCluster {run_id: $run})
    RETURN k.type AS type, count(k) AS count
    """
    type_res, _, _ = driver.execute_query(type_query, run=resolved_run, database_=database)
    types_clustered = {r["type"]: int(r["count"]) for r in type_res if r.get("type")}

    logger.info(
        "Created %d exact-name clusters (%d members linked, largest: %d, run_id: %s)",
        clusters,
        links,
        max_size,
        resolved_run,
    )

    return ClusterSummary(
        run_id=resolved_run,
        clusters_created=clusters,
        members_linked=links,
        largest_cluster_size=max_size,
        types_clustered=types_clustered,
    )


def remove_entity_clusters(
    driver: Driver,
    run_id: str | None = None,
    database: str = "neo4j",
) -> int:
    """Detach and delete EntityCluster nodes for a specific run_id or all clusters if run_id is None.

    Args:
        driver: Active Neo4j driver.
        run_id: Optional identifier of the clustering pass to undo. If None, removes all clusters.
        database: Target Neo4j database name.

    Returns:
        Number of clusters deleted.
    """
    if run_id is not None:
        delete_query = """
        MATCH (k:EntityCluster {run_id: $run})
        DETACH DELETE k
        RETURN count(k) AS deleted
        """
        res, _, _ = driver.execute_query(delete_query, run=run_id, database_=database)
    else:
        delete_query = """
        MATCH (k:EntityCluster)
        DETACH DELETE k
        RETURN count(k) AS deleted
        """
        res, _, _ = driver.execute_query(delete_query, database_=database)
    deleted = res[0]["deleted"] if res else 0
    logger.info("Removed %d entity clusters (run_id=%s)", deleted, run_id)
    return deleted


@contextmanager
def exact_name_clustering(
    driver: Driver,
    run_id: str | None = None,
    database: str = "neo4j",
) -> Iterator[ClusterSummary]:
    """Context manager establishing exact-name clusters and guaranteeing cleanup on exit."""
    summary = create_exact_name_clusters(driver, run_id=run_id, database=database)
    try:
        yield summary
    finally:
        remove_entity_clusters(driver, summary.run_id, database=database)
