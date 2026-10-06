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

    def __init__(self, *, version_rows=None, graph_rows=None, connectivity_error=None):
        self.version_rows = version_rows or [
            {"name": "Neo4j Kernel", "versions": ["5.26.0"], "edition": "community"}
        ]
        self.graph_rows = graph_rows or [{"documents": 1, "chunks": 2}]
        self.connectivity_error = connectivity_error

    def verify_connectivity(self) -> None:
        if self.connectivity_error:
            raise self.connectivity_error

    def connect(self):
        return MagicMock()

    def run_read(self, query: str, parameters=None):
        if "dbms.components" in query:
            return self.version_rows
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
        "properties": ["embedding"],
        "options": {
            "indexConfig": {
                "vector.dimensions": config.get("dimensions", 3),
                "vector.similarity_function": config.get("similarity", "cosine"),
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
