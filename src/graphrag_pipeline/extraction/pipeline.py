"""Build the application's provider-selectable ``SimpleKGPipeline``.

This module assembles GraphRAG's LLM, embedder, and Neo4j driver from provider
settings. It does not load MuSiQue data or run extraction; the
caller supplies prepared text and provenance, then invokes ``run_async``.

The factory configures text input, fail-fast errors, disabled entity resolution,
and the local ``neo4j`` database. Construction initializes the Neo4j writer,
while model and embedding API calls begin only when extraction runs.
"""

from __future__ import annotations

from neo4j import Driver
from neo4j_graphrag.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.embeddings.ollama import OllamaEmbeddings
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import OllamaLLM
from neo4j_graphrag.llm import OpenAILLM

from graphrag_pipeline.config import OllamaSettings, OpenRouterSettings, model_settings_from_env
from graphrag_pipeline.extraction.debug_llm import DebuggableOpenAILLM


def build_simple_kg_pipeline(
    driver: Driver,
    settings: OpenRouterSettings | OllamaSettings | None = None,
) -> SimpleKGPipeline:
    """Assemble, but do not run, the configured extraction pipeline.

    Args:
        driver: The Neo4j Python driver for the target database. The GraphRAG
            writer uses it during construction to check the Neo4j version and
            later uses it to persist the generated graph.
        settings: Optional provider settings instance. When omitted, the
            function selects a provider using ``GRAPH_RAG_PROVIDER``.

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
        llm = DebuggableOpenAILLM(
            model_name=settings.llm_model,
            model_params={
                # "temperature": 0, # -- Must be disabled for some providers which do not support temperature
                "reasoning_effort": "none", # -- This can be toggled depending on provider
                "max_tokens": settings.llm_max_tokens,
                "extra_body": {
                    "provider": {
                        "require_parameters": settings.require_parameters,
                        "allow_fallbacks": settings.allow_fallbacks,
                    },
                },
            },
            base_url=settings.base_url,
            api_key=settings.api_key,
        )
        embedder = OpenAIEmbeddings(model=settings.embedding_model, **client_kwargs)
    else:
        llm = OllamaLLM(
            model_name=settings.llm_model,
            model_params={
                "think": False, # -- Does not need to always be active. Only turn on if using a hybrid model like qwen3:8b
                "options": {"temperature": 0}
                },
            host=settings.host,
        )
        embedder = OllamaEmbeddings(model=settings.embedding_model, host=settings.host)
        schema = "FREE"
    if isinstance(settings, OpenRouterSettings):
        schema = None
    text_splitter = FixedSizeSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,
        schema=schema,
        text_splitter=text_splitter,
        from_file=False,
        on_error="RAISE",
        perform_entity_resolution=False,
        neo4j_database="neo4j",
    )