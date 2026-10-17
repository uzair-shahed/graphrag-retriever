"""Deterministic tests for Neo4j GraphRAG prerequisite verification."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from graphrag_pipeline.config import Neo4jSettings
from graphrag_pipeline.retrieval_prerequisites import (
    RetrievalPrerequisiteError,
    VectorIndexSettings,
    verify_retrieval_prerequisites,
)


class DummyConnection:
    """Neo4j connection double for read-only prerequisite checks."""

    def __init__(self, *, version_rows=None, graph_rows=None, connectivity_error=None, record_rows=None):
        self.version_rows = version_rows or [
            {"name": "Neo4j Kernel", "versions": ["5.26.0"], "edition": "community"}
        ]
        self.graph_rows = graph_rows or [{"documents": 1, "chunks": 2}]
        self.connectivity_error = connectivity_error
        self.record_rows = record_rows or {}
        self.record_queries: list[tuple[str, dict]] = []

    def verify_connectivity(self) -> None:
        if self.connectivity_error:
            raise self.connectivity_error

    def connect(self):
        return MagicMock()

    def run_read(self, query: str, parameters=None):
        if "dbms.components" in query:
            return self.version_rows
        if "chunks_with_property" in query:
            self.record_queries.append((query, parameters or {}))
            return [self.record_rows.get(parameters["record_id"], {"chunks": 0, "chunks_with_property": 0})]
        if "MATCH (d:Document)" in query:
            return self.graph_rows
        raise AssertionError(f"Unexpected query: {query}")


def neo4j_settings() -> Neo4jSettings:
    """Return deterministic connection settings without reading the environment."""
    return Neo4jSettings(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="secret",
        database="neo4j",
        connection_timeout_seconds=1.0,
        max_transaction_retry_time_seconds=1.0,
        transaction_timeout_seconds=1.0,
    )


def index_metadata(**config):
    """Build vector-index metadata in the shape returned by Neo4j."""
    return {
        "name": "chunk_embedding",
        "state": "ONLINE",
        "labelsOrTypes": ["Chunk"],
        "properties": config.get("properties", ["embedding"]),
        "options": {
            "indexConfig": {
                "vector.dimensions": config.get("dimensions", 3),
                "vector.similarity_function": config.get("similarity", "cosine"),
                **config.get("extra_config", {}),
            }
        },
    }


def test_verify_retrieval_prerequisites_returns_report() -> None:
    """A connected graph and matching vector index produce a verified report."""
    connection = DummyConnection()
    model = SimpleNamespace(
        llm_model="test-llm",
        embedding_model="test-embedding",
        embedding_dimensions=3,
    )
    with patch(
        "graphrag_pipeline.retrieval_prerequisites.retrieve_vector_index_info",
        return_value=index_metadata(),
    ):
        report = verify_retrieval_prerequisites(
            connection,
            neo4j_settings(),
            model_settings=model,
            vector_index=VectorIndexSettings(dimensions=3),
        )

    assert report.verified
    assert report.neo4j["database"] == "neo4j"
    assert report.vector_index["dimensions"] == 3
    assert report.graphrag["llm_model"] == "test-llm"
    assert report.graphrag["version"] == "1.21.0"


def test_verify_retrieval_prerequisites_rejects_missing_index() -> None:
    """A missing vector index is an explicit prerequisite failure."""
    with patch(
        "graphrag_pipeline.retrieval_prerequisites.retrieve_vector_index_info",
        return_value=None,
    ):
        with pytest.raises(RetrievalPrerequisiteError, match="was not found"):
            verify_retrieval_prerequisites(DummyConnection(), neo4j_settings())


def test_verify_retrieval_prerequisites_rejects_dimension_mismatch() -> None:
    """An index with incompatible embedding dimensions cannot be used safely."""
    with patch(
        "graphrag_pipeline.retrieval_prerequisites.retrieve_vector_index_info",
        return_value=index_metadata(dimensions=4),
    ):
        with pytest.raises(RetrievalPrerequisiteError, match="dimensions mismatch"):
            verify_retrieval_prerequisites(
                DummyConnection(),
                neo4j_settings(),
                vector_index=VectorIndexSettings(dimensions=3),
            )


def test_verify_retrieval_prerequisites_rejects_unavailable_database() -> None:
    """Connectivity failures do not become a valid empty retrieval state."""
    with pytest.raises(RetrievalPrerequisiteError, match="prerequisite check failed"):
        verify_retrieval_prerequisites(
            DummyConnection(connectivity_error=RuntimeError("database unavailable")),
            neo4j_settings(),
        )


def test_verify_retrieval_prerequisites_rejects_empty_graph() -> None:
    """A reachable but empty graph cannot satisfy retrieval prerequisites."""
    connection = DummyConnection(graph_rows=[{"documents": 0, "chunks": 0}])
    with pytest.raises(RetrievalPrerequisiteError, match="no persisted"):
        verify_retrieval_prerequisites(connection, neo4j_settings())


def test_verify_retrieval_prerequisites_requires_paired_provenance_ids() -> None:
    """Selected provenance verification requires both chunk and record IDs."""
    with pytest.raises(ValueError, match="supplied together"):
        verify_retrieval_prerequisites(
            DummyConnection(), neo4j_settings(), chunk_id="chunk-1"
        )


def verify_with_index(connection, metadata, **kwargs):
    """Run prerequisite verification against fixed vector-index metadata."""
    with patch(
        "graphrag_pipeline.retrieval_prerequisites.retrieve_vector_index_info",
        return_value=metadata,
    ):
        return verify_retrieval_prerequisites(
            connection,
            neo4j_settings(),
            vector_index=VectorIndexSettings(dimensions=3),
            **kwargs,
        )


def test_record_filter_supported_reports_in_index_state_and_record_coverage() -> None:
    """A declared filter property with fully covered chunks is reported as in-index."""
    connection = DummyConnection(record_rows={"record-1": {"chunks": 4, "chunks_with_property": 4}})

    report = verify_with_index(
        connection,
        index_metadata(properties=["embedding", "record_id"]),
        require_record_filter=True,
        record_ids=["record-1"],
    )

    assert report.vector_index["filterable_properties"] == ["record_id"]
    assert report.record_filter == {
        "property": "record_id",
        "index_declares_property": True,
        "state": "in_index",
        "records": {"record-1": {"chunks": 4, "chunks_with_property": 4}},
    }
    assert connection.record_queries[0][1] == {"record_id": "record-1"}


def test_record_filter_declared_through_index_config_key_is_recognized() -> None:
    """The documented vector.filterable_properties config key is also honored."""
    connection = DummyConnection(record_rows={"record-1": {"chunks": 1, "chunks_with_property": 1}})

    report = verify_with_index(
        connection,
        index_metadata(extra_config={"vector.filterable_properties": ["record_id"]}),
        require_record_filter=True,
        record_ids=["record-1"],
    )

    assert report.record_filter["state"] == "in_index"


def test_record_filter_missing_from_index_fails_when_required() -> None:
    """A record-constrained query cannot proceed without an index-declared filter."""
    with pytest.raises(RetrievalPrerequisiteError, match="declared filterable.*vector_index_setup"):
        verify_with_index(DummyConnection(), index_metadata(), require_record_filter=True)


def test_record_filter_not_required_reports_missing_declaration_without_failing() -> None:
    """Unconstrained queries still see whether the index could support a record filter."""
    report = verify_with_index(DummyConnection(), index_metadata())

    assert report.record_filter["state"] == "not_required"
    assert report.record_filter["index_declares_property"] is False
    assert report.vector_index["filterable_properties"] == []


def test_record_filter_fails_when_some_chunks_lack_the_property() -> None:
    """Chunks missing record_id would be silently dropped by the filter, so this fails early."""
    connection = DummyConnection(record_rows={"record-1": {"chunks": 5, "chunks_with_property": 3}})

    with pytest.raises(RetrievalPrerequisiteError, match="2 of 5 chunks.*lack 'record_id'"):
        verify_with_index(
            connection,
            index_metadata(properties=["embedding", "record_id"]),
            require_record_filter=True,
            record_ids=["record-1"],
        )


def test_record_filter_fails_for_record_without_chunks() -> None:
    """A requested record that is not loaded is an explicit prerequisite failure."""
    with pytest.raises(RetrievalPrerequisiteError, match="no persisted chunks"):
        verify_with_index(
            DummyConnection(),
            index_metadata(properties=["embedding", "record_id"]),
            require_record_filter=True,
            record_ids=["missing-record"],
        )


def test_record_filter_rejects_unsafe_label_identifier() -> None:
    """The label is interpolated into Cypher, so only plain identifiers are accepted."""
    label = "Chunk) DETACH DELETE (x"
    metadata = index_metadata(properties=["embedding", "record_id"]) | {"labelsOrTypes": [label]}
    with patch(
        "graphrag_pipeline.retrieval_prerequisites.retrieve_vector_index_info",
        return_value=metadata,
    ):
        with pytest.raises(RetrievalPrerequisiteError, match="simple identifiers"):
            verify_retrieval_prerequisites(
                DummyConnection(),
                neo4j_settings(),
                vector_index=VectorIndexSettings(dimensions=3, label=label),
                require_record_filter=True,
            )
