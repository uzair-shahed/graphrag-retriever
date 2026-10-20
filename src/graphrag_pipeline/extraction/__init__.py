"""Contract 2 extraction models and provider boundary.

Last modified: 2026-09-23

Purpose of this file:
    Re-export the Contract 2 schemas and provider boundary as the public
    extraction package surface.

Files that use this file:
    Extraction agents and tests can import the public models and providers
    from ``graphrag_pipeline.extraction``.

Caveats:
    The extraction agent, retries, dead-letter workflow, LangGraph wiring, and
    external LLM provider are not implemented by this package initializer.
"""

from graphrag_pipeline.extraction.provider import (
    DeterministicMockProvider,
    ExtractionProvider,
)

from graphrag_pipeline.extraction.schema_v1 import (
    DOMAIN_NODE_LABELS,
    DOMAIN_RELATIONSHIP_TYPES,
    RESERVED_NODE_LABELS,
    RESERVED_RELATIONSHIP_TYPES,
    SCHEMA_V1,
    SCHEMA_V1_HASH,
    build_schema_v1,
    compute_schema_hash,
    validate_schema_no_reserved_names,
)
from graphrag_pipeline.extraction.schemas import (
    EvidenceSpan,
    ExtractionRequest,
    ExtractionResult,
    Mention,
    Provenance,
    Triple,
    validate_result_for_request,
)

__all__ = [
    "DOMAIN_NODE_LABELS",
    "DOMAIN_RELATIONSHIP_TYPES",
    "EvidenceSpan",
    "DeterministicMockProvider",
    "ExtractionProvider",
    "ExtractionRequest",
    "ExtractionResult",
    "Mention",
    "Provenance",
    "RESERVED_NODE_LABELS",
    "RESERVED_RELATIONSHIP_TYPES",
    "SCHEMA_V1",
    "SCHEMA_V1_HASH",
    "Triple",
    "build_schema_v1",
    "compute_schema_hash",
    "validate_result_for_request",
    "validate_schema_no_reserved_names",
]