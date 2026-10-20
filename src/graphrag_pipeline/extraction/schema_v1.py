"""Curated fixed extraction schema v1.

Defines fixed domain entity and relationship types for GraphRAG extraction.
Enforces strict pruning of lexical graph names (Document, Chunk, etc.)
and computes a deterministic schema hash.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from neo4j_graphrag.components.schema import (
    GraphSchema,
    NodeType,
    PropertyType,
    RelationshipType,
)

# Reserved labels that must NEVER appear as extracted domain entities or relationships.
# These names collide with neo4j-graphrag's lexical graph model.
RESERVED_NODE_LABELS: frozenset[str] = frozenset({"document", "chunk"})
RESERVED_RELATIONSHIP_TYPES: frozenset[str] = frozenset(
    {"from_chunk", "from_document", "next_chunk", "part_of_chunk"}
)

DOMAIN_NODE_LABELS: tuple[str, ...] = (
    "Person",
    "Organization",
    "Location",
    "Event",
    "CreativeWork",
    "Date",
    "Concept",
)

DOMAIN_RELATIONSHIP_TYPES: tuple[str, ...] = (
    "BORN_IN",
    "DIED_IN",
    "LOCATED_IN",
    "PART_OF",
    "MEMBER_OF",
    "EMPLOYED_BY",
    "FOUNDED_BY",
    "CREATED_BY",
    "PERFORMED_BY",
    "HEADQUARTERED_IN",
    "CAPITAL_OF",
    "SPOUSE_OF",
    "PARENT_OF",
    "CHILD_OF",
    "OCCURRED_IN",
    "OCCURRED_ON",
    "AWARDED",
    "SUCCEEDED",
    "ACQUIRED",
    "RELATED_TO",
)


def validate_schema_no_reserved_names(schema: GraphSchema) -> None:
    """Validate that a GraphSchema does not contain any reserved lexical graph names."""
    for node_type in schema.node_types:
        if node_type.label.lower() in RESERVED_NODE_LABELS:
            raise ValueError(
                f"Schema contains reserved node label '{node_type.label}'. "
                f"Reserved names {sorted(RESERVED_NODE_LABELS)} collide with lexical graph."
            )
    for rel_type in schema.relationship_types:
        if rel_type.label.lower() in RESERVED_RELATIONSHIP_TYPES:
            raise ValueError(
                f"Schema contains reserved relationship type '{rel_type.label}'. "
                f"Reserved names {sorted(RESERVED_RELATIONSHIP_TYPES)} collide with lexical graph."
            )


def compute_schema_hash(schema: GraphSchema) -> str:
    """Compute a deterministic SHA-256 hash for a GraphSchema."""
    canonical_repr: dict[str, Any] = {
        "node_types": sorted(
            [
                {
                    "label": nt.label,
                    "properties": sorted([p.name for p in (nt.properties or [])]),
                }
                for nt in schema.node_types
            ],
            key=lambda x: x["label"],
        ),
        "relationship_types": sorted([rt.label for rt in schema.relationship_types]),
        "additional_node_types": bool(schema.additional_node_types),
        "additional_relationship_types": bool(schema.additional_relationship_types),
    }
    encoded = json.dumps(canonical_repr, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"schema_v1_{hashlib.sha256(encoded).hexdigest()[:16]}"


def build_schema_v1() -> GraphSchema:
    """Build the curated domain schema v1 with strict pruning."""
    node_types = tuple(
        NodeType(
            label=label,
            properties=[PropertyType(name="name", type="STRING", required=False)],
            additional_properties=False,
        )
        for label in DOMAIN_NODE_LABELS
    )
    relationship_types = tuple(
        RelationshipType(
            label=rel,
            properties=[],
            additional_properties=True,
        )
        for rel in DOMAIN_RELATIONSHIP_TYPES
    )
    schema = GraphSchema(
        node_types=node_types,
        relationship_types=relationship_types,
        additional_node_types=False,
        additional_relationship_types=False,
    )
    validate_schema_no_reserved_names(schema)
    return schema


SCHEMA_V1: GraphSchema = build_schema_v1()
SCHEMA_V1_HASH: str = compute_schema_hash(SCHEMA_V1)
