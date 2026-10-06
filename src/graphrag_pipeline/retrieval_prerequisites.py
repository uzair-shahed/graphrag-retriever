"""Verify Neo4j GraphRAG and vector-retrieval prerequisites.

Last modified: 2026-10-06

Purpose of this file:
    Check the read-only prerequisites required before GraphRAG retrieval:
    Neo4j connectivity/version, installed GraphRAG API, vector-index
    configuration, and plugin-created graph content.

Files that use this file:
    The future retrieval command will run ``verify_retrieval_prerequisites``
    before constructing a GraphRAG retriever. Unit tests mock the Neo4j and
    vector-index boundaries.

Caveats:
    This module does not create indexes, ingest data, retrieve evidence, or
    generate answers. It reports the installed API and validates an existing
    index; live Neo4j state is required for a complete report.
"""

from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import asdict, dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Literal

from neo4j_graphrag.generation.graphrag import GraphRAG
from neo4j_graphrag.indexes import retrieve_vector_index_info
from neo4j_graphrag.retrievers import VectorCypherRetriever, VectorRetriever

from graphrag_pipeline.config import Neo4jSettings
from graphrag_pipeline.verification import verify_graph_provenance


class RetrievalPrerequisiteError(RuntimeError):
    """Raised when the graph cannot safely support GraphRAG retrieval."""


@dataclass(frozen=True)
class VectorIndexSettings:
    """Expected Neo4j vector-index contract for GraphRAG retrieval."""

    name: str = "chunk_embedding"
    label: str = "Chunk"
    embedding_property: str = "embedding"
    dimensions: int | None = None
    similarity: Literal["cosine", "euclidean"] = "cosine"

    @classmethod
    def from_env(cls) -> "VectorIndexSettings":
        """Read index names and expected dimensions from environment settings."""
        import os

        raw_dimensions = os.environ.get("GRAPH_RAG_VECTOR_DIMENSIONS")
        dimensions = int(raw_dimensions) if raw_dimensions else None
        if dimensions is not None and dimensions <= 0:
            raise ValueError("GRAPH_RAG_VECTOR_DIMENSIONS must be positive")
        similarity = os.environ.get("GRAPH_RAG_VECTOR_SIMILARITY", "cosine").lower()
        if similarity not in {"cosine", "euclidean"}:
            raise ValueError("GRAPH_RAG_VECTOR_SIMILARITY must be cosine or euclidean")
        return cls(
            name=os.environ.get("GRAPH_RAG_VECTOR_INDEX_NAME", "chunk_embedding"),
            label=os.environ.get("GRAPH_RAG_VECTOR_INDEX_LABEL", "Chunk"),
            embedding_property=os.environ.get(
                "GRAPH_RAG_VECTOR_PROPERTY", "embedding"
            ),
            dimensions=dimensions,
            similarity=similarity,
        )


@dataclass(frozen=True)
class RetrievalPrerequisiteReport:
    """Machine-readable result of prerequisite verification."""

    verified: bool
    neo4j: dict[str, Any]
    graphrag: dict[str, Any]
    vector_index: dict[str, Any]
    graph: dict[str, Any]
    provenance: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a JSON-serializable dictionary."""
        return asdict(self)


def _record_to_dict(record: Any) -> dict[str, Any]:
    """Convert a Neo4j record or test double into a plain dictionary."""
    if record is None:
        return {}
    if isinstance(record, dict):
        return dict(record)
    if hasattr(record, "data"):
        return dict(record.data())
    try:
        return {key: record[key] for key in record.keys()}
    except (AttributeError, KeyError, TypeError) as error:
        raise RetrievalPrerequisiteError("vector index metadata is malformed") from error


def _index_config(metadata: dict[str, Any]) -> dict[str, Any]:
    """Extract Neo4j vector index options from SHOW INDEXES metadata."""
    options = metadata.get("options") or {}
    config = options.get("indexConfig", options) if isinstance(options, dict) else {}
    return dict(config) if isinstance(config, dict) else {}


def _api_report() -> dict[str, Any]:
    """Describe the installed GraphRAG API without making a provider call."""
    try:
        package_version = version("neo4j-graphrag")
    except PackageNotFoundError:
        package_version = "unknown"
    return {
        "package": "neo4j-graphrag",
        "version": package_version,
        "graph_rag_search": str(inspect.signature(GraphRAG.search)),
        "vector_retriever": str(inspect.signature(VectorRetriever)),
        "vector_cypher_retriever": str(inspect.signature(VectorCypherRetriever)),
    }


def _verify_index(
    connection: Any,
    neo4j_settings: Neo4jSettings,
    settings: VectorIndexSettings,
) -> dict[str, Any]:
    """Validate the configured vector index and its dimensions."""
    try:
        metadata = retrieve_vector_index_info(
            connection.connect(),
            settings.name,
            settings.label,
            settings.embedding_property,
            neo4j_database=neo4j_settings.database,
        )
    except Exception as error:  # noqa: BLE001 - normalize driver errors at this boundary
        raise RetrievalPrerequisiteError(f"vector index inspection failed: {error}") from error
    if metadata is None:
        raise RetrievalPrerequisiteError(
            f"vector index {settings.name!r} was not found for "
            f"{settings.label}.{settings.embedding_property}"
        )

    values = _record_to_dict(metadata)
    labels = values.get("labelsOrTypes") or []
    properties = values.get("properties") or []
    if settings.label not in labels or settings.embedding_property not in properties:
        raise RetrievalPrerequisiteError(
            "vector index target does not match the configured Chunk.embedding contract"
        )
    config = _index_config(values)
    dimensions = config.get("vector.dimensions")
    similarity = config.get("vector.similarity_function")
    if settings.dimensions is not None and dimensions != settings.dimensions:
        raise RetrievalPrerequisiteError(
            f"vector index dimensions mismatch: expected {settings.dimensions}, got {dimensions}"
        )
    if similarity is not None and similarity.lower() != settings.similarity:
        raise RetrievalPrerequisiteError(
            f"vector index similarity mismatch: expected {settings.similarity}, got {similarity}"
        )
    state = values.get("state")
    if state is not None and state != "ONLINE":
        raise RetrievalPrerequisiteError(f"vector index {settings.name!r} is not ONLINE: {state}")
    return {
        "name": values.get("name", settings.name),
        "state": state or "reported_by_neo4j",
        "label": settings.label,
        "embedding_property": settings.embedding_property,
        "dimensions": dimensions,
        "similarity": similarity or settings.similarity,
    }


def verify_retrieval_prerequisites(
    connection: Any,
    neo4j_settings: Neo4jSettings,
    *,
    model_settings: Any | None = None,
    vector_index: VectorIndexSettings | None = None,
    chunk_id: str | None = None,
    record_id: str | None = None,
) -> RetrievalPrerequisiteReport:
    """Verify Neo4j, GraphRAG, vector-index, graph, and optional provenance state.

    Args:
        connection: Neo4jConnection-compatible object with read methods.
        neo4j_settings: Connection settings, including the target database.
        model_settings: Optional provider settings to include in the report.
        vector_index: Expected vector-index contract; loaded from environment when omitted.
        chunk_id: Optional persisted chunk to verify read-only.
        record_id: Optional MuSiQue record associated with ``chunk_id``.

    Returns:
        A machine-readable prerequisite report.

    Raises:
        RetrievalPrerequisiteError: If any required capability or graph state is unavailable.
        ValueError: If only one of ``chunk_id`` and ``record_id`` is supplied.
    """
    if (chunk_id is None) != (record_id is None):
        raise ValueError("chunk_id and record_id must be supplied together")
    settings = vector_index or VectorIndexSettings.from_env()
    provider_dimensions = getattr(model_settings, "embedding_dimensions", None)
    if settings.dimensions is None and provider_dimensions is not None:
        settings = replace(settings, dimensions=provider_dimensions)
    try:
        connection.verify_connectivity()
        version_rows = connection.run_read(
            "CALL dbms.components() YIELD name, versions, edition "
            "RETURN name, versions, edition"
        )
        if not version_rows:
            raise RetrievalPrerequisiteError("Neo4j version information is unavailable")
        graph_rows = connection.run_read(
            "MATCH (d:Document) WITH count(d) AS documents "
            "MATCH (c:Chunk) WITH documents, count(c) AS chunks "
            "OPTIONAL MATCH (e:__Entity__) WITH documents, chunks, count(e) AS entities "
            "OPTIONAL MATCH ()-[r]->() RETURN documents, chunks, entities, count(r) AS relationships"
        )
        if not graph_rows or graph_rows[0].get("documents", 0) < 1 or graph_rows[0].get("chunks", 0) < 1:
            raise RetrievalPrerequisiteError("Neo4j graph has no persisted Document and Chunk nodes")
    except RetrievalPrerequisiteError:
        raise
    except Exception as error:  # noqa: BLE001 - normalize database errors
        raise RetrievalPrerequisiteError(f"Neo4j prerequisite check failed: {error}") from error

    index_report = _verify_index(connection, neo4j_settings, settings)
    provenance = None
    if chunk_id is not None and record_id is not None:
        try:
            provenance = verify_graph_provenance(connection, chunk_id, record_id)
        except Exception as error:  # noqa: BLE001 - expose a clear prerequisite failure
            raise RetrievalPrerequisiteError(f"selected graph provenance failed: {error}") from error

    model_report = {"provider": type(model_settings).__name__ if model_settings else "unknown"}
    if model_settings is not None:
        for field_name in (
            "llm_model",
            "embedding_model",
            "embedding_dimensions",
            "host",
            "base_url",
        ):
            value = getattr(model_settings, field_name, None)
            if value is not None:
                model_report[field_name] = value
    return RetrievalPrerequisiteReport(
        verified=True,
        neo4j={"database": neo4j_settings.database, "server": version_rows[0]},
        graphrag=_api_report() | model_report,
        vector_index=index_report,
        graph=graph_rows[0],
        provenance=provenance,
    )


def main() -> None:
    """Run prerequisite verification and print its JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-id", help="Optional persisted Chunk ID to verify.")
    parser.add_argument("--record-id", help="Optional MuSiQue record ID for --chunk-id.")
    args = parser.parse_args()

    from graphrag_pipeline.config import model_settings_from_env
    from graphrag_pipeline.storage.neo4j_connection import Neo4jConnection

    settings = Neo4jSettings.from_env()
    connection = Neo4jConnection(settings)
    try:
        report = verify_retrieval_prerequisites(
            connection,
            settings,
            model_settings=model_settings_from_env(),
            chunk_id=args.chunk_id,
            record_id=args.record_id,
        )
    except (RetrievalPrerequisiteError, ValueError) as error:
        print(json.dumps({"verified": False, "error": str(error)}, indent=2, sort_keys=True))
        raise SystemExit(1) from error
    finally:
        connection.close()
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
