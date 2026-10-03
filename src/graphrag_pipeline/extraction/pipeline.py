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
from neo4j_graphrag.embeddings.ollama import OllamaEmbeddings
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import OllamaLLM
from neo4j_graphrag.llm import OpenAILLM

from graphrag_pipeline.config import OllamaSettings, OpenRouterSettings, model_settings_from_env


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
        llm = OpenAILLM(
            model_name=settings.llm_model,
            model_params={"temperature": 0, "max_tokens": settings.llm_max_tokens},
            base_url=settings.base_url,
            api_key=settings.api_key,
        )
        embedder = OpenAIEmbeddings(model=settings.embedding_model, **client_kwargs)
    else:
        llm = OllamaLLM(
            model_name=settings.llm_model,
            model_params={"options": {"temperature": 0}},
            host=settings.host,
        )
        embedder = OllamaEmbeddings(model=settings.embedding_model, host=settings.host)
        schema = "FREE"
    if isinstance(settings, OpenRouterSettings):
        schema = None
    return SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,
        schema=schema,
        from_file=False,
        on_error="RAISE",
        perform_entity_resolution=False,
        neo4j_database="neo4j",
    )