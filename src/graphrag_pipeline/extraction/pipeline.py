"""Build the application's provider-selectable ``SimpleKGPipeline``.

This module assembles GraphRAG's LLM, embedder, and Neo4j driver from provider
settings. It does not load MuSiQue data or run extraction; the
caller supplies prepared text and provenance, then invokes ``run_async``.

The factory configures full-paragraph text input, fail-fast errors, disabled
entity resolution and the atomic source-scoped writer. Construction is offline;
database setup and provider calls begin only when ingestion runs.
"""

from __future__ import annotations

from neo4j import Driver
from neo4j_graphrag.components.types import Neo4jGraph
from neo4j_graphrag.embeddings.ollama import OllamaEmbeddings
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

from graphrag_pipeline.config import Neo4jSettings, OllamaSettings, OpenRouterSettings, model_settings_from_env
from graphrag_pipeline.extraction.atomic_writer import AtomicKGWriter
from graphrag_pipeline.extraction.idempotency import ParagraphSplitter
from graphrag_pipeline.extraction.debug_llm import DebuggableOpenAILLM
from graphrag_pipeline.ollama_json_schema import OllamaJSONSchemaLLM


def build_simple_kg_pipeline(
    driver: Driver,
    settings: OpenRouterSettings | OllamaSettings | None = None,
    neo4j_settings: Neo4jSettings | None = None,
) -> SimpleKGPipeline:
    """Assemble, but do not run, the configured extraction pipeline.

    Args:
        driver: Borrowed Neo4j driver used to persist the generated graph.
        settings: Optional provider settings instance. When omitted, the
            function selects a provider using ``GRAPH_RAG_PROVIDER``.
        neo4j_settings: Database and bounded transaction settings. Defaults to
            database ``neo4j`` and the standard transaction limits when omitted.

    Returns:
        A configured ``SimpleKGPipeline`` using the selected GraphRAG provider.

    Raises:
        ValueError: If the selected provider settings are missing or invalid.
        neo4j_graphrag.experimental.pipeline.exceptions.PipelineDefinitionError:
            If the installed GraphRAG version rejects the component contract
            or custom components cannot initialize.

    Notes:
        The atomic writer creates/verifies its scoped constraints on write.
        Configured splitter size/overlap do not split validated paragraphs.
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
        llm = OllamaJSONSchemaLLM(
            model_name=settings.llm_model,
            response_model=Neo4jGraph,
            model_params={"options": {"temperature": 0}},
            host=settings.host,
        )
        embedder = OllamaEmbeddings(model=settings.embedding_model, host=settings.host)
        schema = "FREE"
    if isinstance(settings, OpenRouterSettings):
        # A fixed free schema avoids per-run automatic schema generation and
        # conflicting constraints when multiple paragraphs are preloaded.
        schema = "FREE"
    text_splitter = ParagraphSplitter()
    return SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,
        schema=schema,
        text_splitter=text_splitter,
        kg_writer=AtomicKGWriter(driver, neo4j_settings),
        from_file=False,
        on_error="RAISE",
        perform_entity_resolution=False,
        neo4j_database=neo4j_settings.database if neo4j_settings else "neo4j",
    )
