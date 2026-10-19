"""Run the focused MuSiQue paragraph through SimpleKGPipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from graphrag_pipeline.config import Neo4jSettings, model_settings_from_env
from graphrag_pipeline.diagnostics import (
    exception_diagnostics,
    save_invalid_llm_response,
)
from graphrag_pipeline.extraction.pipeline import build_simple_kg_pipeline
from graphrag_pipeline.extraction.atomic_writer import IngestionStateError
from graphrag_pipeline.extraction.ingest import ingest_paragraph
from graphrag_pipeline.ingestion.retry_policy import attempt_configuration, classify_failure
from graphrag_pipeline.ingestion.retry_queue import RetryStore
from graphrag_pipeline.ingestion.reconcile import COMPLETE
from graphrag_pipeline.extraction.schemas import ExtractionRequest
from graphrag_pipeline.storage.neo4j_connection import Neo4jConnection

DEFAULT_FIXTURE = Path("tests/fixtures/musique/extraction_chunk.json")


def load_fixture(path: Path) -> ExtractionRequest:
    """Load and validate one model-visible MuSiQue paragraph fixture."""
    return ExtractionRequest.model_validate(json.loads(path.read_text(encoding="utf-8")))


async def run_fixture(path: Path) -> dict[str, Any]:
    """Persist one validated fixture paragraph through the live pipeline.

    The Neo4j connection remains open until the asynchronous pipeline completes.
    No graph reset is performed; generated data remains in the configured
    database for inspection.
    """
    request = load_fixture(path)
    queue = RetryStore().for_request(request)
    queue.ensure_request(request)
    try:
        neo4j_settings = Neo4jSettings.from_env()
        model_settings = model_settings_from_env()
    except Exception as error:
        queue.record_observation(request.chunk_id, classify_failure(error, "configuration"), "configuration")
        raise
    connection = Neo4jConnection(neo4j_settings)
    try:
        pipeline = None

        def get_pipeline():
            nonlocal pipeline
            pipeline = build_simple_kg_pipeline(connection.connect(), model_settings, neo4j_settings=neo4j_settings)
            return pipeline

        try:
            ingested = await ingest_paragraph(connection, None, request, retry_queue=queue,
                configuration=attempt_configuration(neo4j_settings, model_settings), pipeline_factory=get_pipeline)
            if ingested.state.state != COMPLETE:
                raise IngestionStateError(f"{ingested.state.state}: {ingested.state.reason}")
        except Exception as error:  # noqa: BLE001 - preserve GraphRAG failures
            if pipeline is not None:
                try:
                    llm = pipeline.runner.config.llm_config["default"].root
                    response_path = save_invalid_llm_response(error, getattr(llm, "last_response", None))
                    if response_path is not None:
                        print(json.dumps({"invalid_llm_response": response_path.as_posix()}))
                except Exception as diagnostic_error:
                    print(json.dumps({"diagnostic_error": exception_diagnostics(diagnostic_error)}))
            raise
        return {
            "status": "succeeded",
            "run_id": ingested.result.run_id if ingested.result is not None else None,
            "reused": ingested.reused,
            "fixture": str(path),
            "chunk_id": request.chunk_id,
            "record_id": request.record_id,
            "database": neo4j_settings.database,
            "provider": type(model_settings).__name__.removesuffix("Settings").lower(),
            "llm_model": model_settings.llm_model,
            "embedding_model": model_settings.embedding_model,
            "result_type": type(ingested.result.result).__name__ if ingested.result is not None else None,
        }
    finally:
        connection.close()


def main() -> None:
    """Run the focused fixture and print a redacted execution summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=DEFAULT_FIXTURE,
        help="Path to one validated extraction chunk JSON file.",
    )
    args = parser.parse_args()
    try:
        summary = asyncio.run(run_fixture(args.fixture))
    except Exception as error:  # noqa: BLE001 - CLI reports the complete cause chain
        print(
            json.dumps(exception_diagnostics(error), indent=2, sort_keys=True),
        )
        raise SystemExit(1) from error
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
