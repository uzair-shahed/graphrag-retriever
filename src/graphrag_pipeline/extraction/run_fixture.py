"""Run the focused MuSiQue paragraph through SimpleKGPipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from graphrag_pipeline.config import Neo4jSettings, OpenRouterSettings
from graphrag_pipeline.extraction.pipeline import build_simple_kg_pipeline
from graphrag_pipeline.extraction.plugin_input import build_plugin_input
from graphrag_pipeline.extraction.schemas import ExtractionRequest
from graphrag_pipeline.storage.neo4j_connection import Neo4jConnection

DEFAULT_FIXTURE = Path("tests/fixtures/musique/extraction_chunk.json")


def load_fixture(path: Path) -> ExtractionRequest:
    """Load and validate one model-visible MuSiQue paragraph fixture."""
    return ExtractionRequest.model_validate(json.loads(path.read_text()))


async def run_fixture(path: Path) -> dict[str, Any]:
    """Persist one validated fixture paragraph through the live pipeline.

    The Neo4j connection remains open until the asynchronous pipeline completes.
    No graph reset is performed; generated data remains in the configured
    database for inspection.
    """
    request = load_fixture(path)
    text, document_metadata = build_plugin_input(request)
    neo4j_settings = Neo4jSettings.from_env()
    openrouter_settings = OpenRouterSettings.from_env()
    connection = Neo4jConnection(neo4j_settings)
    try:
        pipeline = build_simple_kg_pipeline(
            connection.connect(), openrouter_settings
        )
        result = await pipeline.run_async(
            text=text,
            document_metadata=document_metadata,
        )
        return {
            "status": "succeeded",
            "run_id": result.run_id,
            "fixture": str(path),
            "chunk_id": request.chunk_id,
            "record_id": request.record_id,
            "database": neo4j_settings.database,
            "llm_model": openrouter_settings.llm_model,
            "embedding_model": openrouter_settings.embedding_model,
            "embedding_dimensions": openrouter_settings.embedding_dimensions,
            "result_type": type(result.result).__name__,
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
    summary = asyncio.run(run_fixture(args.fixture))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
