"""Integration tests for record-filtered vector search against real Neo4j.

Builds an isolated fixture (its own labels, index names, and ``run_id``) with
two "records" whose chunk vectors are arranged so the *other* record is always
the nearest neighbour. Searching with a record filter must still return only
the requested record's chunks, through an index that declares ``record_id``
filterable. Skips when Neo4j is unreachable/unconfigured or the server cannot
create filterable vector indexes (Neo4j 2026.01+). Fixture nodes and indexes are
removed in a ``finally`` block and never touch the real ``chunk_embedding`` index.
"""

from __future__ import annotations

import time
import uuid

import pytest
from neo4j.exceptions import ServiceUnavailable
from neo4j_graphrag.exceptions import Neo4jIndexError
from neo4j_graphrag.indexes import create_vector_index

from graphrag_pipeline.config import Neo4jSettings
from graphrag_pipeline.retrieval import (
    RecordFilterRetriever,
    RetrievalError,
    _format_record,
    _retrieval_query_vector,
)
from graphrag_pipeline.storage.neo4j_connection import Neo4jConnection

QUERY_VECTOR = [1.0, 0.0, 0.0]
# Record B is always closer to the query than record A.
CHUNKS = [
    ("a1", "record-a", [0.6, 0.8, 0.0]),
    ("a2", "record-a", [0.5, 0.87, 0.0]),
    ("b1", "record-b", [1.0, 0.0, 0.0]),
    ("b2", "record-b", [0.99, 0.1, 0.0]),
]


def _connection_or_skip() -> Neo4jConnection:
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


def _build_fixture(connection: Neo4jConnection, run_id: str, label: str, index_name: str, filterable: list[str]):
    """Create fixture chunks/documents and a vector index; skip if unsupported."""
    for name, record_id, vector in CHUNKS:
        connection.run_write(
            f"CREATE (c:`{label}` {{run_id: $run_id, text: $text, record_id: $record_id, embedding: $vector}}) "
            "CREATE (c)-[:FROM_DOCUMENT]->(:Document {run_id: $run_id, chunk_id: $chunk_id, "
            "record_id: $record_id, document_id: $chunk_id})",
            {
                "run_id": run_id,
                "text": f"text {name}",
                "record_id": record_id,
                "vector": vector,
                "chunk_id": f"fixture:{run_id}:{name}",
            },
        )
    try:
        create_vector_index(
            connection.connect(),
            index_name,
            label,
            "embedding",
            3,
            "cosine",
            fail_if_exists=True,
            filterable_properties=filterable or None,
        )
    except Neo4jIndexError as error:
        pytest.skip(f"server cannot create this vector index: {error}")
    for _ in range(60):
        rows = connection.run_read(
            "SHOW INDEXES YIELD name, state WHERE name = $name RETURN state", {"name": index_name}
        )
        if rows and rows[0]["state"] == "ONLINE":
            return
        time.sleep(0.25)
    pytest.skip("fixture vector index did not become ONLINE")


def _cleanup(connection: Neo4jConnection, run_id: str, index_names: list[str]) -> None:
    for name in index_names:
        connection.run_write("DROP INDEX $name IF EXISTS", {"name": name})
    connection.run_write("MATCH (n) WHERE n.run_id = $run_id DETACH DELETE n", {"run_id": run_id})


def _retriever(connection: Neo4jConnection, index_name: str) -> RecordFilterRetriever:
    return RecordFilterRetriever(
        driver=connection.connect(),
        index_name=index_name,
        retrieval_query=_retrieval_query_vector(),
        embedder=None,
        result_formatter=_format_record,
        neo4j_database=Neo4jSettings.from_env().database,
    )


def _suffixes(result) -> list[str]:
    return [item.metadata["chunk_id"].rsplit(":", 1)[1] for item in result.items]


def test_record_filter_uses_the_index_and_excludes_nearer_unrelated_records() -> None:
    """The nearest chunks belong to another record, yet the filtered result is only record A."""
    connection = _connection_or_skip()
    run_id = uuid.uuid4().hex
    label, index_name = f"FixtureChunk{run_id[:8]}", f"fixture_filterable_{run_id[:8]}"
    try:
        _build_fixture(connection, run_id, label, index_name, ["record_id"])
        retriever = _retriever(connection, index_name)

        unfiltered = retriever.search(query_vector=QUERY_VECTOR, top_k=2)
        filtered = retriever.search(query_vector=QUERY_VECTOR, top_k=2, filters={"record_id": "record-a"})

        assert set(_suffixes(unfiltered)) == {"b1", "b2"}
        assert set(_suffixes(filtered)) == {"a1", "a2"}
        assert {item.metadata["record_id"] for item in filtered.items} == {"record-a"}
    finally:
        _cleanup(connection, run_id, [index_name])
        connection.close()


def test_filtered_top_k_is_filled_from_the_requested_record_only() -> None:
    """A top_k larger than the record's chunk count returns just that record, never other records."""
    connection = _connection_or_skip()
    run_id = uuid.uuid4().hex
    label, index_name = f"FixtureChunk{run_id[:8]}", f"fixture_filterable_{run_id[:8]}"
    try:
        _build_fixture(connection, run_id, label, index_name, ["record_id"])
        filtered = _retriever(connection, index_name).search(
            query_vector=QUERY_VECTOR, top_k=10, filters={"record_id": "record-b"}
        )

        assert sorted(_suffixes(filtered)) == ["b1", "b2"]
    finally:
        _cleanup(connection, run_id, [index_name])
        connection.close()


def test_record_filter_on_index_without_declaration_fails_instead_of_falling_back() -> None:
    """An index that does not declare record_id never silently filters by brute force."""
    connection = _connection_or_skip()
    run_id = uuid.uuid4().hex
    label, index_name = f"FixtureChunk{run_id[:8]}", f"fixture_plain_{run_id[:8]}"
    try:
        _build_fixture(connection, run_id, label, index_name, [])
        retriever = _retriever(connection, index_name)

        with pytest.raises(RetrievalError, match="does not declare"):
            retriever.search(query_vector=QUERY_VECTOR, top_k=2, filters={"record_id": "record-a"})
        assert len(retriever.search(query_vector=QUERY_VECTOR, top_k=2).items) == 2
    finally:
        _cleanup(connection, run_id, [index_name])
        connection.close()
