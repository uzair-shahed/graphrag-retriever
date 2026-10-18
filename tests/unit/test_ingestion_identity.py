"""Source identity, legacy handling and graph-integrity checks without services."""

from unittest.mock import MagicMock

import pytest

from graphrag_pipeline.extraction.idempotency import (
    CONTRACT_VERSION, MANAGED_LABEL, ParagraphSplitter, canonical_json, ensure_schema, fingerprint, managed_key, request_json,
)
from graphrag_pipeline.extraction.plugin_input import build_plugin_input
from graphrag_pipeline.ingestion.reconcile import ABSENT, COMPLETE, CONFLICT, LEGACY, PARTIAL, classify_chunk
from test_reconcile import make_request, row


@pytest.mark.parametrize("field", ["text", "title", "record_id", "document_id", "chunk_id", "paragraph_idx"])
def test_identity_includes_all_request_fields(field):
    request = make_request()
    changed = request.model_copy(update={field: 8 if field == "paragraph_idx" else "changed"})
    assert fingerprint(request) != fingerprint(changed)


@pytest.mark.parametrize("field", ["source_dataset", "source_split", "source_file", "source_sha256", "source_line"])
def test_identity_includes_all_provenance(field):
    request = make_request()
    provenance = request.provenance.model_copy(update={field: 8 if field == "source_line" else "changed"})
    assert fingerprint(request) != fingerprint(request.model_copy(update={"provenance": provenance}))


def test_identity_roundtrip_and_no_unicode_or_whitespace_normalization():
    request = make_request().model_copy(update={"text": "  café\n\t"})
    assert request_json(request) == request_json(type(request).model_validate_json(request.model_dump_json()))
    assert fingerprint(request) != fingerprint(request.model_copy(update={"text": request.text.strip()}))
    assert fingerprint(request) != fingerprint(request.model_copy(update={"text": "  cafe\u0301\n\t"}))


@pytest.mark.parametrize("text", ["  paragraph\n\t", "a" * 30000])
def test_splitter_preserves_whole_paragraph(text):
    chunks = list(ParagraphSplitter().iter_chunks(text))
    assert len(chunks) == 1 and chunks[0].text == text and chunks[0].index == 0


def test_missing_title_is_valid_at_plugin_boundary():
    request = make_request().model_copy(update={"title": None})
    assert "title" not in build_plugin_input(request)[1]
    assert fingerprint(request) != fingerprint(make_request())


def classified(lexical, marker=None, nodes=None, relationships=None):
    connection = MagicMock()
    actual_nodes = nodes if nodes is not None else (managed_source_nodes() if marker is not None else [])
    connection.run_read.side_effect = [lexical, [] if marker is None else [{"completion": marker}], actual_nodes, relationships or [], []]
    return classify_chunk(connection, make_request())


def managed_source_nodes():
    return [{"key": managed_key(make_request().chunk_id, role, "source"), "labels": sorted([label, MANAGED_LABEL]), "properties": {}}
            for role, label in (("document", "Document"), ("chunk", "Chunk"))]


def marker(**overrides):
    return {"fingerprint": fingerprint(make_request()), "request_json": request_json(make_request()),
            "contract_version": CONTRACT_VERSION, "inventory_json": canonical_json({"nodes": sorted(managed_source_nodes(), key=canonical_json), "relationships": []}), **overrides}


def test_legacy_matching_source_is_not_complete():
    assert classified([row()]).state == LEGACY


def test_absent_requires_no_managed_or_completion_state():
    assert classified([]).state == ABSENT
    assert classified([], nodes=[{"key": "orphan", "labels": [], "properties": {}}]).state == PARTIAL
    assert classified([], marker()).state == PARTIAL


def test_complete_requires_matching_marker_and_current_inventory():
    assert classified([row()], marker()).state == COMPLETE
    assert classified([row()], marker(fingerprint="wrong")).state == CONFLICT
    assert classified([row()], marker(request_json="wrong")).state == CONFLICT
    assert classified([row()], marker(contract_version=99)).state == PARTIAL
    assert classified([row()], marker(inventory_json="wrong")).state == PARTIAL


def test_added_graph_relationship_invalidates_completion():
    state = classified([row()], marker(), relationships=[{"key": "unexpected"}])
    assert state.state == PARTIAL


def test_marker_cannot_certify_unmanaged_legacy_source():
    assert classified([row()], marker(inventory_json=canonical_json({"nodes": [], "relationships": []})), nodes=[]).state == PARTIAL


def test_schema_name_collision_is_not_silently_accepted():
    connection = MagicMock()
    connection.run_read.return_value = [{"type": "UNIQUENESS", "entityType": "NODE", "labelsOrTypes": ["Wrong"], "properties": ["wrong"]}]
    with pytest.raises(RuntimeError, match="incompatible ingestion constraint"):
        ensure_schema(connection)
