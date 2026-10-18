"""Make the Chunk vector index filterable by ``record_id`` for MuSiQue queries.

Last modified: 2026-10-17

Purpose of this file:
    Report and, with ``--apply``, repair the two conditions that
    record-constrained vector search needs: every ``Chunk`` carries the
    ``record_id`` of its ``Document``, and the vector index declares
    ``record_id`` as a filterable property (Neo4j 2026.01+ ``SEARCH`` clause,
    ``CREATE VECTOR INDEX ... WITH [n.record_id]``).

Files that use this file:
    Run as ``python -m graphrag_pipeline.vector_index_setup``. Its report fields
    match what ``graphrag_pipeline.retrieval_prerequisites`` verifies before a
    record-constrained query. Unit tests use a fake connection.

Caveats:
    ``--apply`` drops and recreates the index (Neo4j cannot alter filterable
    properties in place), so vector search is unavailable until the new index is
    ONLINE; embeddings are not recomputed. It preserves the existing index
    dimensions and similarity function. Without ``--apply`` nothing is changed.
    Newly ingested records receive ``Chunk.record_id`` inside the atomic writer;
    only graphs loaded earlier need the backfill.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from typing import Any

from neo4j_graphrag.indexes import create_vector_index

from graphrag_pipeline.config import Neo4jSettings
from graphrag_pipeline.retrieval_prerequisites import VectorIndexSettings
from graphrag_pipeline.storage.neo4j_connection import Neo4jConnection

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class VectorIndexSetupError(RuntimeError):
    """Raised when the index cannot be inspected, recreated, or brought ONLINE."""


def _safe_identifier(value: str, what: str) -> str:
    """Return ``value`` if it is a plain Cypher identifier, else raise.

    Labels and property names are interpolated into Cypher, so only simple
    identifiers from configuration are accepted.
    """
    if not _IDENTIFIER.match(value):
        raise VectorIndexSetupError(f"{what} is not a safe Cypher identifier: {value!r}")
    return value


def read_index(connection: Any, name: str) -> dict[str, Any] | None:
    """Read one vector index's definition.

    Args:
        connection: Object with ``run_read``.
        name: Vector index name.

    Returns:
        ``None`` when the index does not exist, otherwise a dict with ``state``,
        ``dimensions``, ``similarity``, and ``filterable_properties`` (a list).
    """
    rows = connection.run_read(
        "SHOW VECTOR INDEXES YIELD name, state, properties, options WHERE name = $name "
        "RETURN state, properties, options.indexConfig AS config",
        {"name": name},
    )
    if not rows:
        return None
    config = rows[0].get("config") or {}
    # Neo4j 2026.x lists filterable properties after the vector property.
    indexed = list(rows[0].get("properties") or [])[1:]
    return {
        "state": rows[0].get("state"),
        "dimensions": config.get("vector.dimensions"),
        "similarity": str(config.get("vector.similarity_function", "")).lower() or None,
        "filterable_properties": sorted({*indexed, *(config.get("vector.filterable_properties") or [])}),
    }


def count_chunks_missing_property(connection: Any, label: str, prop: str) -> int:
    """Count chunks whose Document has ``prop`` but the chunk itself does not."""
    label, prop = _safe_identifier(label, "label"), _safe_identifier(prop, "property")
    rows = connection.run_read(
        f"MATCH (c:`{label}`)-[:FROM_DOCUMENT]->(d:Document) "
        f"WHERE c.`{prop}` IS NULL AND d.`{prop}` IS NOT NULL "
        "RETURN count(c) AS missing"
    )
    return int(rows[0]["missing"]) if rows else 0


def backfill_chunk_property(connection: Any, label: str, prop: str) -> int:
    """Copy ``prop`` from each Document onto its Chunk where the chunk lacks it.

    Args:
        connection: Object with ``run_write``.
        label: Indexed chunk label.
        prop: Property to copy (``record_id``).

    Returns:
        The number of chunks updated; repeated calls return 0.
    """
    label, prop = _safe_identifier(label, "label"), _safe_identifier(prop, "property")
    rows = connection.run_write(
        f"MATCH (c:`{label}`)-[:FROM_DOCUMENT]->(d:Document) "
        f"WHERE c.`{prop}` IS NULL AND d.`{prop}` IS NOT NULL "
        f"SET c.`{prop}` = d.`{prop}` "
        "RETURN count(c) AS updated"
    )
    return int(rows[0]["updated"]) if rows else 0


def _wait_until_online(connection: Any, name: str, timeout_seconds: float) -> None:
    """Poll until the index is ONLINE or raise after ``timeout_seconds``."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        index = read_index(connection, name)
        if index and index["state"] == "ONLINE":
            return
        if time.monotonic() >= deadline:
            raise VectorIndexSetupError(f"vector index {name!r} did not become ONLINE in time")
        time.sleep(0.5)


def ensure_record_filter_index(
    connection: Any,
    neo4j_settings: Neo4jSettings,
    settings: VectorIndexSettings,
    *,
    apply: bool = False,
    online_timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """Report or repair record-filter support for the configured vector index.

    Args:
        connection: ``Neo4jConnection``-compatible object (``run_read``,
            ``run_write``, ``connect``).
        neo4j_settings: Provides the target database name.
        settings: Expected index name, label, embedding property, and filter property.
        apply: When false, only report; when true, backfill ``Chunk`` properties
            and recreate the index if it is not yet filterable.
        online_timeout_seconds: How long to wait for the recreated index.

    Returns:
        A summary with ``action`` (``none``, ``would_recreate``, ``recreated``),
        the filterable properties before and after, and backfill counts.

    Raises:
        VectorIndexSetupError: If the index is missing, or cannot be recreated.
    """
    prop = _safe_identifier(settings.filter_property, "filter property")
    index = read_index(connection, settings.name)
    if index is None:
        raise VectorIndexSetupError(
            f"vector index {settings.name!r} was not found; create it first (see README)"
        )
    missing_before = count_chunks_missing_property(connection, settings.label, prop)
    summary: dict[str, Any] = {
        "index": settings.name,
        "filter_property": prop,
        "filterable_properties_before": index["filterable_properties"],
        "chunks_missing_property_before": missing_before,
        "applied": apply,
    }
    needs_index = prop not in index["filterable_properties"]
    if not apply:
        summary["action"] = "would_recreate" if needs_index else "none"
        summary["would_backfill_chunks"] = missing_before
        return summary

    summary["chunks_backfilled"] = backfill_chunk_property(connection, settings.label, prop)
    if not needs_index:
        summary["action"] = "none"
        summary["filterable_properties_after"] = index["filterable_properties"]
        return summary
    if not index["dimensions"] or index["similarity"] not in {"cosine", "euclidean"}:
        raise VectorIndexSetupError("existing index dimensions or similarity are unreadable")
    filterable = sorted({*index["filterable_properties"], prop})
    connection.run_write("DROP INDEX $name IF EXISTS", {"name": settings.name})
    create_vector_index(
        connection.connect(),
        settings.name,
        settings.label,
        settings.embedding_property,
        int(index["dimensions"]),
        index["similarity"],
        fail_if_exists=True,
        neo4j_database=neo4j_settings.database,
        filterable_properties=filterable,
    )
    _wait_until_online(connection, settings.name, online_timeout_seconds)
    summary["action"] = "recreated"
    summary["filterable_properties_after"] = read_index(connection, settings.name)["filterable_properties"]
    return summary


def main() -> None:
    """Print the record-filter report; with ``--apply`` repair the graph and index."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Backfill Chunk.record_id and recreate the index.")
    args = parser.parse_args()
    neo4j_settings = Neo4jSettings.from_env()
    connection = Neo4jConnection(neo4j_settings)
    try:
        summary = ensure_record_filter_index(
            connection, neo4j_settings, VectorIndexSettings.from_env(), apply=args.apply
        )
    except VectorIndexSetupError as error:
        print(json.dumps({"error": str(error)}, indent=2, sort_keys=True))
        raise SystemExit(1) from error
    finally:
        connection.close()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
