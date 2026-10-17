"""Deterministic tests for the record-filterable vector index setup command."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from graphrag_pipeline.retrieval_prerequisites import VectorIndexSettings
from graphrag_pipeline.vector_index_setup import (
    VectorIndexSetupError,
    backfill_chunk_property,
    ensure_record_filter_index,
    read_index,
)


class FakeConnection:
    """Scripted Neo4j connection recording writes and serving index/chunk state."""

    def __init__(self, *, filterable=(), missing_chunks=2, index_exists=True):
        self.filterable = list(filterable)
        self.missing_chunks = missing_chunks
        self.index_exists = index_exists
        self.writes: list[str] = []

    def connect(self):
        return "driver"

    def run_read(self, query, parameters=None):
        if "SHOW VECTOR INDEXES" in query:
            if not self.index_exists:
                return []
            config = {"vector.dimensions": 768, "vector.similarity_function": "COSINE"}
            return [{"state": "ONLINE", "properties": ["embedding", *self.filterable], "config": config}]
        if "RETURN count(c) AS missing" in query:
            return [{"missing": self.missing_chunks}]
        raise AssertionError(query)

    def run_write(self, query, parameters=None):
        self.writes.append(query)
        if "RETURN count(c) AS updated" in query:
            updated, self.missing_chunks = self.missing_chunks, 0
            return [{"updated": updated}]
        if query.startswith("DROP INDEX"):
            self.index_exists = False
        return []


def neo4j_settings():
    return SimpleNamespace(database="neo4j")


def test_read_index_parses_dimensions_similarity_and_filterable_properties() -> None:
    """The index definition is exposed in a normalized shape."""
    index = read_index(FakeConnection(filterable=["record_id"]), "chunk_embedding")

    assert index == {
        "state": "ONLINE",
        "dimensions": 768,
        "similarity": "cosine",
        "filterable_properties": ["record_id"],
    }


def test_read_index_also_honors_the_filterable_properties_config_key() -> None:
    """Older servers that report the documented config key are still understood."""
    connection = MagicMock()
    connection.run_read.return_value = [
        {
            "state": "ONLINE",
            "properties": ["embedding"],
            "config": {"vector.filterable_properties": ["record_id"]},
        }
    ]

    assert read_index(connection, "chunk_embedding")["filterable_properties"] == ["record_id"]


def test_read_index_returns_none_for_missing_index() -> None:
    """A missing index is distinguishable from an unfilterable one."""
    assert read_index(FakeConnection(index_exists=False), "chunk_embedding") is None


def test_dry_run_reports_planned_changes_without_writing() -> None:
    """Without apply, nothing is modified and the plan is reported."""
    connection = FakeConnection()

    summary = ensure_record_filter_index(connection, neo4j_settings(), VectorIndexSettings())

    assert summary["action"] == "would_recreate"
    assert summary["would_backfill_chunks"] == 2
    assert summary["applied"] is False
    assert connection.writes == []


def test_apply_backfills_then_recreates_index_preserving_dimensions_and_similarity() -> None:
    """Apply copies record_id onto chunks and recreates the index with the filter property."""
    connection = FakeConnection()
    states = iter([{"filterable_properties": ["record_id"]}])

    with (
        patch("graphrag_pipeline.vector_index_setup.create_vector_index") as create,
        patch("graphrag_pipeline.vector_index_setup._wait_until_online"),
        patch(
            "graphrag_pipeline.vector_index_setup.read_index",
            side_effect=[
                {"state": "ONLINE", "dimensions": 768, "similarity": "cosine", "filterable_properties": []},
                next(states),
            ],
        ),
    ):
        summary = ensure_record_filter_index(
            connection, neo4j_settings(), VectorIndexSettings(), apply=True
        )

    assert summary["action"] == "recreated"
    assert summary["chunks_backfilled"] == 2
    assert summary["filterable_properties_after"] == ["record_id"]
    assert connection.writes[0].startswith("MATCH (c:`Chunk`)")
    assert connection.writes[1].startswith("DROP INDEX")
    args, kwargs = create.call_args
    assert args == ("driver", "chunk_embedding", "Chunk", "embedding", 768, "cosine")
    assert kwargs["filterable_properties"] == ["record_id"]
    assert kwargs["fail_if_exists"] is True


def test_apply_does_not_recreate_an_already_filterable_index() -> None:
    """Re-running apply on a configured index only backfills missing chunks."""
    connection = FakeConnection(filterable=["record_id"], missing_chunks=0)

    with patch("graphrag_pipeline.vector_index_setup.create_vector_index") as create:
        summary = ensure_record_filter_index(
            connection, neo4j_settings(), VectorIndexSettings(), apply=True
        )

    assert summary["action"] == "none"
    assert summary["chunks_backfilled"] == 0
    create.assert_not_called()
    assert not any(write.startswith("DROP INDEX") for write in connection.writes)


def test_missing_index_is_an_explicit_error() -> None:
    """The command never invents an index with guessed dimensions."""
    with pytest.raises(VectorIndexSetupError, match="was not found"):
        ensure_record_filter_index(
            FakeConnection(index_exists=False), neo4j_settings(), VectorIndexSettings(), apply=True
        )


def test_unsafe_identifiers_are_rejected_before_any_cypher_runs() -> None:
    """Label and property are interpolated into Cypher, so they must be plain identifiers."""
    connection = MagicMock()

    with pytest.raises(VectorIndexSetupError, match="safe Cypher identifier"):
        backfill_chunk_property(connection, "Chunk`) DETACH DELETE (x", "record_id")
    with pytest.raises(VectorIndexSetupError, match="safe Cypher identifier"):
        backfill_chunk_property(connection, "Chunk", "record_id = 1 //")

    connection.run_write.assert_not_called()
