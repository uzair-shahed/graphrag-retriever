"""Tests for the environment-driven ``SimpleKGPipeline`` configuration.

These tests cover only local configuration and construction policy. They do
not call OpenRouter, ingest a MuSiQue paragraph, or write application data to
Neo4j. The pipeline test uses a real Neo4j ``Driver`` instance because the
installed GraphRAG API validates that type. Construction of the atomic writer
does not access the database.
"""

from unittest.mock import patch

from neo4j import GraphDatabase
from neo4j_graphrag.components.types import Neo4jGraph

from graphrag_pipeline.config import OllamaSettings, OpenRouterSettings
from graphrag_pipeline.extraction.pipeline import build_simple_kg_pipeline
from graphrag_pipeline.extraction.idempotency import ParagraphSplitter
from graphrag_pipeline.extraction.atomic_writer import AtomicKGWriter
from graphrag_pipeline.ollama_json_schema import OllamaJSONSchemaLLM


def openrouter_environment() -> dict[str, str]:
    """Return a complete, secret-free environment fixture for OpenRouter."""
    return {
        "OPENROUTER_API_KEY": "test-key",
        "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
        "OPENROUTER_LLM_MODEL": "openai/gpt-4o-mini",
        "OPENROUTER_LLM_MAX_TOKENS": "4096",
        "OPENROUTER_EMBEDDING_MODEL": "openai/text-embedding-3-small",
        "OPENROUTER_EMBEDDING_DIMENSIONS": "1536",
        "OPENROUTER_REQUIRE_PARAMETERS": "true",
        "OPENROUTER_ALLOW_FALLBACKS": "true",
    }


def test_openrouter_settings_parse_without_exposing_secret() -> None:
    """Verify model and dimension settings parse without logging credentials."""
    with patch.dict("os.environ", openrouter_environment(), clear=False):
        settings = OpenRouterSettings.from_env()

    assert settings.llm_model == "openai/gpt-4o-mini"
    assert settings.llm_max_tokens == 4096
    assert settings.embedding_model == "openai/text-embedding-3-small"
    assert settings.embedding_dimensions == 1536
    assert settings.api_key == "test-key"


def test_pipeline_factory_uses_minimal_fixture_configuration() -> None:
    """Verify provider policy, atomic writer, paragraph splitter and disabled resolver offline."""
    settings = OpenRouterSettings(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        llm_model="openai/gpt-4o-mini",
        embedding_model="openai/text-embedding-3-small",
        embedding_dimensions=1536,
        llm_max_tokens=2048,
        require_parameters=True,
        allow_fallbacks=True,
    )

    driver = GraphDatabase.driver("bolt://127.0.0.1:7687", auth=("test", "test"))
    try:
        pipeline = build_simple_kg_pipeline(driver=driver, settings=settings)

        llm = pipeline.runner.config.llm_config["default"].root
        assert llm.model_name == "openai/gpt-4o-mini"
        assert llm.model_params["extra_body"]["provider"] == {
            "require_parameters": True,
            "allow_fallbacks": True,
        }
        assert pipeline.runner.config.from_file is False
        assert pipeline.runner.config.on_error.value == "RAISE"
        assert pipeline.runner.config.perform_entity_resolution is False
        assert pipeline.runner.config.neo4j_database == "neo4j"
        splitter = pipeline.runner.config.text_splitter.root
        assert isinstance(splitter, ParagraphSplitter)
        assert isinstance(pipeline.runner.config.kg_writer.root, AtomicKGWriter)
    finally:
        driver.close()


def test_pipeline_factory_builds_ollama_components() -> None:
    """Verify Ollama settings create native GraphRAG Ollama components offline."""
    settings = OllamaSettings(
        host="http://localhost:11434",
        llm_model="qwen3:4b",
        embedding_model="nomic-embed-text",
    )

    driver = GraphDatabase.driver("bolt://127.0.0.1:7687", auth=("test", "test"))
    try:
        pipeline = build_simple_kg_pipeline(driver=driver, settings=settings)

        llm = pipeline.runner.config.llm_config["default"].root
        embedder = pipeline.runner.config.embedder_config["default"].root
        assert isinstance(llm, OllamaJSONSchemaLLM)
        assert llm.model_name == "qwen3:4b"
        assert llm.response_schema == Neo4jGraph.model_json_schema()
        assert llm.supports_structured_output is True
        assert llm.model_params["options"]["temperature"] == 0
        assert type(embedder).__name__ == "OllamaEmbeddings"
        assert embedder.model == "nomic-embed-text"
        assert embedder.client._client.base_url == "http://localhost:11434"
        assert pipeline.runner.config.from_file is False
        assert pipeline.runner.config.on_error.value == "RAISE"
        assert pipeline.runner.config.perform_entity_resolution is False
        assert pipeline.runner.config.neo4j_database == "neo4j"
    finally:
        driver.close()
