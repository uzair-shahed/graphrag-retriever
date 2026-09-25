"""Tests for read-only graph/provenance verification after a plugin run."""

import pytest

from graphrag_pipeline.verification import verify_graph_provenance


class DummyConnection:
    def __init__(self, *, document_rows, chunk_rows, relationship_rows):
        self.document_rows = document_rows
        self.chunk_rows = chunk_rows
        self.relationship_rows = relationship_rows

    def run_read(self, query: str, parameters: dict | None = None):
        if "MATCH (d:Document" in query:
            return self.document_rows
        if "MATCH (c:Chunk" in query:
            return self.chunk_rows
        if "MATCH ()-[r]->()" in query:
            return self.relationship_rows
        raise AssertionError(f"Unexpected query: {query}")


def test_verify_graph_provenance_accepts_valid_fixture_graph() -> None:
    connection = DummyConnection(
        document_rows=[
            {
                "document_id": "musique:document:aster-systems-acquisition",
                "chunk_id": "musique:fixture_valid_multi_hop_001:paragraph:1",
                "record_id": "fixture_valid_multi_hop_001",
                "paragraph_idx": "1",
                "source_dataset": "musique",
                "source_split": "fixture",
                "source_file": "tests/fixtures/musique/valid_records.jsonl",
                "source_sha256": "fixture-sha256",
                "source_line": "1",
                "title": "Aster Systems Acquisition",
            }
        ],
        chunk_rows=[
            {
                "chunk_id": "musique:fixture_valid_multi_hop_001:paragraph:1",
                "text": "Aster Systems acquired Northwind Labs in a strategic deal announced in 2021.",
                "embedding": [0.1, 0.2, 0.3],
            }
        ],
        relationship_rows=[
            {"relationship_type": "FROM_CHUNK", "count": 2},
            {"relationship_type": "FROM_DOCUMENT", "count": 1},
            {"relationship_type": "ACQUIRED", "count": 1},
        ],
    )

    result = verify_graph_provenance(
        connection,
        chunk_id="musique:fixture_valid_multi_hop_001:paragraph:1",
        record_id="fixture_valid_multi_hop_001",
    )

    assert result["verified"] is True
    assert result["document"]["record_id"] == "fixture_valid_multi_hop_001"
    assert result["source_metadata"]["source_file"] == "tests/fixtures/musique/valid_records.jsonl"
    assert result["relationship_counts"]["FROM_CHUNK"] == 2


def test_verify_graph_provenance_rejects_missing_source_metadata() -> None:
    connection = DummyConnection(
        document_rows=[
            {
                "document_id": "musique:document:aster-systems-acquisition",
                "chunk_id": "musique:fixture_valid_multi_hop_001:paragraph:1",
                "record_id": "fixture_valid_multi_hop_001",
                "paragraph_idx": "1",
                "source_dataset": "musique",
                "source_split": "fixture",
                "source_sha256": "fixture-sha256",
                "source_line": "1",
            }
        ],
        chunk_rows=[
            {
                "chunk_id": "musique:fixture_valid_multi_hop_001:paragraph:1",
                "text": "Aster Systems acquired Northwind Labs in a strategic deal announced in 2021.",
                "embedding": [0.1, 0.2, 0.3],
            }
        ],
        relationship_rows=[
            {"relationship_type": "FROM_CHUNK", "count": 2},
        ],
    )

    with pytest.raises(ValueError, match="source_file"):
        verify_graph_provenance(
            connection,
            chunk_id="musique:fixture_valid_multi_hop_001:paragraph:1",
            record_id="fixture_valid_multi_hop_001",
        )
