"""Read-only verification helpers for a successful plugin extraction run."""

from __future__ import annotations

from typing import Any


def _require_fields(record: dict[str, Any], required: list[str], label: str) -> None:
    missing = [field for field in required if field not in record or record[field] in (None, "")]
    if missing:
        raise ValueError(f"Missing {label} provenance fields: {', '.join(missing)}")


def verify_graph_provenance(
    connection: Any,
    chunk_id: str,
    record_id: str,
    expected_relationship_types: set[str] | list[str] | None = None,
) -> dict[str, Any]:
    """Run read-only Neo4j queries and validate the extracted graph against the MuSiQue fixture.

    The function is intentionally small and conservative: it verifies that the
    expected source-backed document and chunk records exist and that the graph
    includes the required relationship types.
    """
    document_rows = connection.run_read(
        """
        MATCH (d:Document)
        WHERE d.chunk_id = $chunk_id OR d.record_id = $record_id
        RETURN d.chunk_id AS chunk_id,
               d.record_id AS record_id,
               d.document_id AS document_id,
               d.paragraph_idx AS paragraph_idx,
               d.source_dataset AS source_dataset,
               d.source_split AS source_split,
               d.source_file AS source_file,
               d.source_sha256 AS source_sha256,
               d.source_line AS source_line,
               d.title AS title
        LIMIT 10
        """,
        {"chunk_id": chunk_id, "record_id": record_id},
    )
    if not document_rows:
        raise ValueError(f"No Document node found for chunk_id={chunk_id!r} or record_id={record_id!r}")

    document = document_rows[0]
    _require_fields(
        document,
        [
            "chunk_id",
            "record_id",
            "paragraph_idx",
            "source_dataset",
            "source_split",
            "source_file",
            "source_sha256",
            "source_line",
        ],
        "document",
    )

    chunk_rows = connection.run_read(
        """
        MATCH (c:Chunk)
        WHERE c.chunk_id = $chunk_id
        RETURN c.chunk_id AS chunk_id, c.text AS text, c.embedding AS embedding
        LIMIT 10
        """,
        {"chunk_id": chunk_id},
    )
    if not chunk_rows:
        raise ValueError(f"No Chunk node found for chunk_id={chunk_id!r}")

    chunk = chunk_rows[0]
    _require_fields(chunk, ["chunk_id", "text"], "chunk")

    relationship_rows = connection.run_read(
        """
        MATCH ()-[r]->()
        RETURN type(r) AS relationship_type, count(r) AS count
        ORDER BY relationship_type
        """
    )
    relationship_counts = {row["relationship_type"]: row["count"] for row in relationship_rows}
    required_types = (
        set(expected_relationship_types)
        if expected_relationship_types is not None
        else {"FROM_CHUNK", "FROM_DOCUMENT"}
    )
    missing = sorted(required_types - set(relationship_counts))
    if missing:
        raise ValueError(f"Missing expected relationship types in graph: {missing}")

    for req in required_types:
        if relationship_counts.get(req, 0) < 1:
            raise ValueError(f"Graph does not contain the expected {req} relationship")

    return {
        "verified": True,
        "document": document,
        "chunk": chunk,
        "source_metadata": {
            "source_dataset": document["source_dataset"],
            "source_split": document["source_split"],
            "source_file": document["source_file"],
            "source_sha256": document["source_sha256"],
            "source_line": document["source_line"],
            "record_id": document["record_id"],
            "chunk_id": document["chunk_id"],
            "paragraph_idx": document["paragraph_idx"],
        },
        "relationship_counts": relationship_counts,
    }
