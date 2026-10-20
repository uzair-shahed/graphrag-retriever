"""Build the application's provider-selectable ``SimpleKGPipeline``.

This module assembles GraphRAG's LLM, embedder, and Neo4j driver from provider
settings. It does not load MuSiQue data or run extraction; the
caller supplies prepared text and provenance, then invokes ``run_async``.

The factory configures text input, fail-fast errors, disabled entity resolution,
and the local ``neo4j`` database. Construction initializes the Neo4j writer,
while model and embedding API calls begin only when extraction runs.
"""

from __future__ import annotations

import logging
from typing import Any

from neo4j import Driver
from neo4j_graphrag.components.schema import GraphSchema
from neo4j_graphrag.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.embeddings.ollama import OllamaEmbeddings
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import OllamaLLM
from neo4j_graphrag.llm import OpenAILLM

from graphrag_pipeline.config import (
    OllamaSettings,
    OpenRouterSettings,
    is_reasoning_model,
    model_settings_from_env,
)
from graphrag_pipeline.extraction.debug_llm import DebuggableOpenAILLM
from graphrag_pipeline.extraction.schema_v1 import build_schema_v1

logger = logging.getLogger(__name__)


class GraphWriteError(RuntimeError):
    """Raised when the Neo4j writer fails to persist the knowledge graph."""


def ensure_writer_succeeded(result: Any) -> None:
    """Ensure that the Neo4j writer component succeeded, raising GraphWriteError if not.

    The neo4j-graphrag library catches writer exceptions internally and records
    `status="FAILURE"` in its metadata dictionary rather than re-raising. This function
    fails closed so failed writes cannot be silently treated as successes.
    """
    writer = (getattr(result, "result", None) or {}).get("writer")
    if not writer or writer.get("status") != "SUCCESS":
        raise GraphWriteError(f"Neo4j writer did not succeed: {writer!r}")


def clear_pipeline_store(pipeline: Any) -> None:
    """Clear in-memory intermediate and final result stores from a SimpleKGPipeline.

    neo4j-graphrag accumulates component outputs in `runner.pipeline.store` and
    `runner.pipeline.final_results` across runs on the same pipeline instance.
    Because finished paragraph runs are never read again, emptying these stores
    bounds memory to O(1) across thousands of dataset records.
    """
    runner = getattr(pipeline, "runner", None)
    inner = getattr(runner, "pipeline", None)
    if inner is not None:
        store = getattr(inner, "store", None)
        if store is not None and hasattr(store, "empty"):
            store.empty()
        final_results = getattr(inner, "final_results", None)
        if final_results is not None and hasattr(final_results, "empty"):
            final_results.empty()



def build_simple_kg_pipeline(
    driver: Driver,
    settings: OpenRouterSettings | OllamaSettings | None = None,
    schema: GraphSchema | None = None,
) -> SimpleKGPipeline:
    """Assemble, but do not run, the configured extraction pipeline.

    Args:
        driver: The Neo4j Python driver for the target database. The GraphRAG
            writer uses it during construction to check the Neo4j version and
            later uses it to persist the generated graph.
        settings: Optional provider settings instance. When omitted, the
            function selects a provider using ``GRAPH_RAG_PROVIDER``.
        schema: Optional GraphSchema to govern extraction. When omitted for
            OpenRouter, defaults to the curated domain ``SCHEMA_V1``.

    Returns:
        A configured ``SimpleKGPipeline`` using the selected GraphRAG provider.

    Raises:
        ValueError: If the selected provider settings are missing or invalid.
        neo4j_graphrag.experimental.pipeline.exceptions.PipelineDefinitionError:
            If the installed GraphRAG version rejects the component contract
            or the Neo4j writer cannot initialize.

    Notes:
        ``SimpleKGPipeline`` performs a Neo4j version check while constructing
        its writer. Model and embedding calls occur only during ``run_async``.
    """
    settings = settings or model_settings_from_env()
    if isinstance(settings, OpenRouterSettings):
        client_kwargs = {"api_key": settings.api_key, "base_url": settings.base_url}
        provider_dict: dict[str, Any] = {
            "require_parameters": settings.require_parameters,
            "allow_fallbacks": settings.allow_fallbacks,
        }
        if settings.provider_order:
            provider_dict["order"] = settings.provider_order

        model_params: dict[str, Any] = {
            "temperature": settings.temperature,
            "max_tokens": settings.llm_max_tokens,
            "extra_body": {
                "provider": provider_dict,
            },
        }
        if settings.seed is not None:
            model_params["seed"] = settings.seed
        if is_reasoning_model(settings.llm_model):
            model_params["reasoning_effort"] = "none"

        logger.info(
            "Configured extraction OpenAILLM: model=%s, profile=%s, temp=%s, seed=%s, provider_order=%s",
            settings.llm_model,
            settings.profile,
            settings.temperature,
            settings.seed,
            settings.provider_order,
        )
        llm = DebuggableOpenAILLM(
            model_name=settings.llm_model,
            model_params=model_params,
            base_url=settings.base_url,
            api_key=settings.api_key,
        )
        embedder = OpenAIEmbeddings(model=settings.embedding_model, **client_kwargs)
        pipeline_schema: GraphSchema | str = schema or build_schema_v1()
    else:
        llm = OllamaLLM(
            model_name=settings.llm_model,
            model_params={"options": {"temperature": 0}},
            host=settings.host,
        )
        embedder = OllamaEmbeddings(model=settings.embedding_model, host=settings.host)
        pipeline_schema = "FREE"
    text_splitter = FixedSizeSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,
        schema=pipeline_schema,
        text_splitter=text_splitter,
        from_file=False,
        on_error="RAISE",
        perform_entity_resolution=False,
        neo4j_database="neo4j",
    )