"""Validate graph structure and identity before opening a write transaction."""

import math

import pytest
from neo4j_graphrag.components.types import LexicalGraphConfig, Neo4jGraph

from graphrag_pipeline.extraction.atomic_writer import prepare_graph
from graphrag_pipeline.extraction.idempotency import managed_key
from graphrag_pipeline.extraction.plugin_input import build_plugin_input
from test_reconcile import make_request


def graph():
    request = make_request()
    _, metadata = build_plugin_input(request)
    return Neo4jGraph.model_validate({
        "nodes": [
            {"id": "doc", "label": "Document", "properties": metadata},
            {"id": "chunk", "label": "Chunk", "properties": {"text": request.text, "index": 0},
             "embedding_properties": {"embedding": [0.25, 0.5]}},
            {"id": "person", "label": "Person", "properties": {"name": "Alex"}},
        ],
        "relationships": [
            {"start_node_id": "chunk", "end_node_id": "doc", "type": "FROM_DOCUMENT"},
            {"start_node_id": "person", "end_node_id": "chunk", "type": "FROM_CHUNK"},
        ],
    })


def test_prepared_graph_copies_full_source_metadata_and_scopes_keys():
    request, nodes, relationships = prepare_graph(graph(), LexicalGraphConfig())
    assert request == make_request()
    source = next(n for n in nodes if "Chunk" in n["labels"])
    assert source["properties"]["source_split"] == request.provenance.source_split
    assert source["properties"]["text"] == request.text
    assert source["properties"]["chunk_id"] == request.chunk_id
    assert len({n["key"] for n in nodes}) == 3
    assert source["key"] == managed_key(request.chunk_id, "chunk", "source")
    assert all(r["properties"]["ingestion_chunk_id"] == request.chunk_id for r in relationships)


@pytest.mark.parametrize("case", [
    "no_document", "no_chunk", "multiple_chunks", "duplicate_node_id", "unknown_endpoint",
    "duplicate_relationship", "missing_entity_source", "wrong_document_source", "reserved_label",
    "reserved_node_property", "reserved_relationship_property", "empty_embedding", "nan_embedding",
    "infinite_property", "domain_to_document", "embedding_overwrite", "custom_lexical_config",
])
def test_malformed_graph_is_rejected(case):
    value = graph()
    config = LexicalGraphConfig()
    if case == "no_document":
        value.nodes.pop(0)
    elif case == "no_chunk":
        value.nodes.pop(1)
    elif case == "multiple_chunks":
        value.nodes.append(value.nodes[1].model_copy(update={"id": "chunk2"}))
    elif case == "duplicate_node_id":
        value.nodes[2].id = "chunk"
    elif case == "unknown_endpoint":
        value.relationships[1].start_node_id = "missing"
    elif case == "duplicate_relationship":
        value.relationships.append(value.relationships[1].model_copy())
    elif case == "missing_entity_source":
        value.relationships.pop()
    elif case == "wrong_document_source":
        value.relationships[0].start_node_id = "person"
    elif case == "reserved_label":
        value.nodes[2].label = "IngestionCompletion"
    elif case == "reserved_node_property":
        value.nodes[2].properties["ingestion_key"] = "spoofed"
    elif case == "reserved_relationship_property":
        value.relationships[0].properties["ingestion_key"] = "spoofed"
    elif case == "empty_embedding":
        value.nodes[1].embedding_properties = {}
    elif case == "nan_embedding":
        value.nodes[1].embedding_properties["embedding"] = [math.nan]
    elif case == "infinite_property":
        value.nodes[2].properties["confidence"] = math.inf
    elif case == "domain_to_document":
        value.relationships[0].type = "FOUNDED"
    elif case == "embedding_overwrite":
        value.nodes[1].embedding_properties["text"] = [1.0]
    elif case == "custom_lexical_config":
        config = LexicalGraphConfig(chunk_node_label="OtherChunk")
    with pytest.raises(ValueError):
        prepare_graph(value, config)
