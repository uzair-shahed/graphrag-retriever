"""Read-only reconciliation of one chunk request against persisted Neo4j state.

The loader consults this module before spending provider calls and after a
write, so that resumed runs trust Neo4j rather than only the local manifest.
Nothing here writes to, repairs, or deletes graph data: ``partial`` and
``conflict`` outcomes are reported for later manual investigation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from graphrag_pipeline.extraction.schemas import ExtractionRequest
from graphrag_pipeline.extraction.idempotency import (
    CONTRACT_VERSION, MANAGED_LABEL, canonical_json, completion_rows, fingerprint, managed_key, read_inventory, request_json,
)

ABSENT = "absent"
COMPLETE = "complete"
PARTIAL = "partial"
CONFLICT = "conflict"
LEGACY = "legacy_unverified"

_CHUNK_STATE_QUERY = """
MATCH (d:Document {chunk_id: $chunk_id})
OPTIONAL MATCH (c:Chunk)-[:FROM_DOCUMENT]->(d)
RETURN elementId(d) AS document_element_id,
       properties(d) AS document,
       collect(c {.text, .chunk_id, .record_id, .document_id, .source_file,
                  .source_line, .source_sha256, .title,
                  .paragraph_idx, element_id: elementId(c)}) AS chunks
"""


class _ReadConnection(Protocol):
    def run_read(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]: ...


@dataclass
class ChunkState:
    """Classification of one chunk request in Neo4j, with locating details."""

    state: str
    reason: str = ""
    document_element_ids: list[str] = field(default_factory=list)
    chunk_element_ids: list[str] = field(default_factory=list)

    def to_report(self, request: ExtractionRequest) -> dict[str, Any]:
        """Return a JSON-safe entry sufficient to locate and clean up the nodes."""
        return {
            "chunk_id": request.chunk_id,
            "record_id": request.record_id,
            "state": self.state,
            "reason": self.reason,
            "document_count": len(self.document_element_ids),
            "chunk_count": len(self.chunk_element_ids),
            "document_element_ids": list(self.document_element_ids),
            "chunk_element_ids": list(self.chunk_element_ids),
        }


def _expected_document_metadata(request: ExtractionRequest) -> dict[str, str | None]:
    provenance = request.provenance
    return {
        "record_id": request.record_id,
        "chunk_id": request.chunk_id,
        "document_id": request.document_id,
        "title": request.title,
        "source_dataset": provenance.source_dataset,
        "source_split": provenance.source_split,
        "source_file": provenance.source_file,
        "source_sha256": provenance.source_sha256,
        "source_line": str(provenance.source_line),
        "paragraph_idx": str(request.paragraph_idx),
    }


def _metadata_matches(actual: Any, expected: str | None) -> bool:
    """Compare string-normalized metadata while allowing an absent title."""
    return actual is None if expected is None else str(actual) == expected


def classify_lexical_chunk(connection: _ReadConnection, request: ExtractionRequest) -> ChunkState:
    """Classify a chunk request as ``absent``, ``complete``, ``partial`` or ``conflict``.

    ``complete`` means exactly one ``Document`` for the ``chunk_id`` whose
    provenance matches the request, with exactly one ``Chunk`` carrying the
    unchanged paragraph text and copied provenance. ``conflict`` means the same
    identity holds different input. ``partial`` means expected graph structure
    is missing, duplicated, or lacks provenance.
    """
    rows = connection.run_read(_CHUNK_STATE_QUERY, {"chunk_id": request.chunk_id})
    if not rows:
        return ChunkState(ABSENT)

    document_ids = [str(row["document_element_id"]) for row in rows]
    if len(rows) > 1:
        chunk_ids = [str(chunk["element_id"]) for row in rows for chunk in row["chunks"]]
        return ChunkState(
            CONFLICT,
            f"{len(rows)} Document nodes share this chunk_id",
            document_ids,
            chunk_ids,
        )

    row = rows[0]
    chunks = row["chunks"]
    chunk_element_ids = [str(chunk["element_id"]) for chunk in chunks]

    document = row["document"] or {}
    for key, expected in _expected_document_metadata(request).items():
        if not _metadata_matches(document.get(key), expected):
            return ChunkState(
                CONFLICT,
                f"Document {key} is {document.get(key)!r}, expected {expected!r}",
                document_ids,
                chunk_element_ids,
            )

    if not chunks:
        return ChunkState(PARTIAL, "Document exists but has no Chunk", document_ids, chunk_element_ids)
    if len(chunks) != 1:
        return ChunkState(
            PARTIAL,
            f"expected exactly 1 Chunk per paragraph, found {len(chunks)}",
            document_ids,
            chunk_element_ids,
        )

    chunk = chunks[0]
    if chunk.get("text") != request.text:
        return ChunkState(
            CONFLICT, "Chunk text differs from the source paragraph", document_ids, chunk_element_ids
        )
    expected_chunk = _expected_document_metadata(request)
    for key in (
        "chunk_id", "record_id", "document_id", "source_file", "source_line",
        "source_sha256", "title", "paragraph_idx",
    ):
        if not _metadata_matches(chunk.get(key), expected_chunk[key]):
            return ChunkState(
                PARTIAL,
                f"Chunk is missing provenance property {key!r}",
                document_ids,
                chunk_element_ids,
            )
    return ChunkState(COMPLETE, "", document_ids, chunk_element_ids)


def classify_chunk(connection: _ReadConnection, request: ExtractionRequest) -> ChunkState:
    """Only an intact versioned atomic write is complete; legacy data stays unverified."""
    lexical = classify_lexical_chunk(connection, request)
    if lexical.state == CONFLICT:
        return lexical
    markers = completion_rows(connection, request.chunk_id)
    inventory = read_inventory(connection, request.chunk_id)
    if not markers:
        if lexical.state == ABSENT and not any(inventory.values()):
            orphans = connection.run_read(
                "MATCH (c:Chunk) WHERE c.chunk_id = $chunk_id "
                "OR (c.text = $text AND NOT (c)-[:FROM_DOCUMENT]->(:Document)) "
                "RETURN elementId(c) AS id, c.text AS text", {"chunk_id": request.chunk_id, "text": request.text},
            )
            if orphans:
                return ChunkState(PARTIAL, "unlinked or legacy Chunk exists without source Document; audit required",
                                  [], [str(row["id"]) for row in orphans])
        if lexical.state == COMPLETE:
            lexical.state = LEGACY
            lexical.reason = "matching legacy source has no atomic completion record; audit required"
        elif lexical.state == ABSENT and any(inventory.values()):
            lexical.state = PARTIAL
            lexical.reason = "managed graph exists without source Document or completion record"
        return lexical
    if len(markers) != 1:
        lexical.state, lexical.reason = CONFLICT, "duplicate completion records"
        return lexical
    marker = markers[0]["completion"]
    if marker.get("fingerprint") != fingerprint(request) or marker.get("request_json") != request_json(request):
        lexical.state, lexical.reason = CONFLICT, "completion record refers to different source input"
    elif lexical.state != COMPLETE:
        lexical.state, lexical.reason = PARTIAL, "completion exists but source graph is incomplete: " + lexical.reason
    elif marker.get("contract_version") != CONTRACT_VERSION:
        lexical.state, lexical.reason = PARTIAL, "unsupported ingestion write-contract version"
    elif any(
        len([node for node in inventory["nodes"]
             if node["key"] == managed_key(request.chunk_id, role, "source")
             and label in node["labels"] and MANAGED_LABEL in node["labels"]]) != 1
        for role, label in (("document", "Document"), ("chunk", "Chunk"))
    ):
        lexical.state, lexical.reason = PARTIAL, "completion lacks uniquely managed source Document/Chunk"
    elif marker.get("inventory_json") != canonical_json(inventory):
        lexical.state, lexical.reason = PARTIAL, "persisted graph inventory differs from atomic completion record"
    return lexical
