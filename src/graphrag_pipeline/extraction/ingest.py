"""Validated ingestion boundary shared by batch, preload and fixture commands."""

from dataclasses import dataclass
from typing import Any

from graphrag_pipeline.extraction.atomic_writer import IngestionStateError
from graphrag_pipeline.extraction.plugin_input import build_plugin_input
from graphrag_pipeline.ingestion.reconcile import ABSENT, COMPLETE, ChunkState, classify_chunk


@dataclass
class IngestionResult:
    state: ChunkState
    result: Any = None
    reused: bool = False


async def ingest_paragraph(connection, pipeline, request, *, before_write=None, classify=classify_chunk) -> IngestionResult:
    """Reconcile before model calls, and verify atomic completion before returning.

    before_write persists the optional local in-flight checkpoint. It is invoked
    only for absent work; the database remains authoritative without a manifest.
    """
    state = classify(connection, request)
    if state.state == COMPLETE:
        return IngestionResult(state, reused=True)
    if state.state != ABSENT:
        return IngestionResult(state)
    if before_write is not None:
        before_write()
    text, metadata = build_plugin_input(request)
    result = await pipeline.run_async(text=text, document_metadata=metadata)
    written = classify(connection, request)
    if written.state != COMPLETE:
        raise IngestionStateError(f"post-write Neo4j verification failed ({written.state}): {written.reason}")
    return IngestionResult(written, result=result)
