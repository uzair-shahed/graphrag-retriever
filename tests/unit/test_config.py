"""Unit tests for `Neo4jSettings` env parsing. No network access required."""

from __future__ import annotations

import pytest

from graphrag_pipeline.config import (
    Neo4jSettings,
    OllamaSettings,
    OpenRouterSettings,
    is_reasoning_model,
    model_settings_from_env,
)


def test_from_env_reads_required_and_default_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "test-password")
    monkeypatch.delenv("NEO4J_DATABASE", raising=False)
    monkeypatch.delenv("NEO4J_CONNECTION_TIMEOUT_SECONDS", raising=False)

    settings = Neo4jSettings.from_env()

    assert settings.uri == "bolt://localhost:7687"
    assert settings.username == "neo4j"
    assert settings.password == "test-password"
    assert settings.database == "neo4j"
    assert settings.connection_timeout_seconds == 5.0
    assert settings.max_transaction_retry_time_seconds == 15.0
    assert settings.transaction_timeout_seconds == 10.0


def test_from_env_reads_overridden_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "test-password")
    monkeypatch.setenv("NEO4J_DATABASE", "fixture-db")
    monkeypatch.setenv("NEO4J_CONNECTION_TIMEOUT_SECONDS", "2.5")
    monkeypatch.setenv("NEO4J_MAX_TRANSACTION_RETRY_TIME_SECONDS", "20")
    monkeypatch.setenv("NEO4J_TRANSACTION_TIMEOUT_SECONDS", "1")

    settings = Neo4jSettings.from_env()

    assert settings.database == "fixture-db"
    assert settings.connection_timeout_seconds == 2.5
    assert settings.max_transaction_retry_time_seconds == 20.0
    assert settings.transaction_timeout_seconds == 1.0


def test_from_env_requires_uri_username_and_password(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEO4J_URI", raising=False)
    monkeypatch.delenv("NEO4J_USERNAME", raising=False)
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)

    with pytest.raises(ValueError):
        Neo4jSettings.from_env()


def test_from_env_rejects_non_numeric_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "test-password")
    monkeypatch.setenv("NEO4J_CONNECTION_TIMEOUT_SECONDS", "not-a-number")

    with pytest.raises(ValueError):
        Neo4jSettings.from_env()


def test_model_settings_from_env_reads_ollama_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Ollama provider supplies the expected local model defaults."""
    monkeypatch.setenv("GRAPH_RAG_PROVIDER", "ollama")
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_LLM_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_EMBEDDING_MODEL", raising=False)

    settings = model_settings_from_env()

    assert isinstance(settings, OllamaSettings)
    assert settings.host == "http://localhost:11434"
    assert settings.llm_model == "qwen3:4b"
    assert settings.embedding_model == "nomic-embed-text"
    assert settings.chunk_size == 10000
    assert settings.chunk_overlap == 0


def test_model_settings_from_env_reads_ollama_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Ollama host and model names can be configured through the environment."""
    monkeypatch.setenv("GRAPH_RAG_PROVIDER", "OLLAMA")
    monkeypatch.setenv("OLLAMA_HOST", "http://ollama.example:11434/")
    monkeypatch.setenv("OLLAMA_LLM_MODEL", "qwen3:8b")
    monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "custom-embedder")

    settings = model_settings_from_env()

    assert settings == OllamaSettings(
        host="http://ollama.example:11434",
        llm_model="qwen3:8b",
        embedding_model="custom-embedder",
    )


def test_model_settings_from_env_reads_chunk_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shared GraphRAG splitter settings can be overridden through the environment."""
    monkeypatch.setenv("GRAPH_RAG_PROVIDER", "ollama")
    monkeypatch.setenv("GRAPH_RAG_CHUNK_SIZE", "12000")
    monkeypatch.setenv("GRAPH_RAG_CHUNK_OVERLAP", "250")

    settings = model_settings_from_env()

    assert isinstance(settings, OllamaSettings)
    assert settings.chunk_size == 12000
    assert settings.chunk_overlap == 250


def test_model_settings_from_env_rejects_invalid_chunk_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chunk overlap must be smaller than chunk size."""
    monkeypatch.setenv("GRAPH_RAG_PROVIDER", "ollama")
    monkeypatch.setenv("GRAPH_RAG_CHUNK_SIZE", "1000")
    monkeypatch.setenv("GRAPH_RAG_CHUNK_OVERLAP", "1000")

    with pytest.raises(ValueError, match="GRAPH_RAG_CHUNK_OVERLAP"):
        model_settings_from_env()


def test_model_settings_from_env_rejects_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unsupported provider values fail before a pipeline is constructed."""
    monkeypatch.setenv("GRAPH_RAG_PROVIDER", "unknown")

    with pytest.raises(ValueError, match="GRAPH_RAG_PROVIDER"):
        model_settings_from_env()


def test_openrouter_settings_default_experiment_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default profile is experiment with temperature 0, seed, pinned provider, and no fallbacks."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENROUTER_LLM_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")
    monkeypatch.setenv("OPENROUTER_EMBEDDING_DIMENSIONS", "1536")
    monkeypatch.delenv("GRAPH_RAG_PROFILE", raising=False)
    monkeypatch.delenv("GRAPH_RAG_TEMPERATURE", raising=False)
    monkeypatch.delenv("GRAPH_RAG_SEED", raising=False)
    monkeypatch.delenv("OPENROUTER_PROVIDER_ORDER", raising=False)
    monkeypatch.delenv("OPENROUTER_ALLOW_FALLBACKS", raising=False)

    settings = OpenRouterSettings.from_env()

    assert settings.profile == "experiment"
    assert settings.temperature == 0.0
    assert settings.seed == 20261018
    assert settings.provider_order == ["openai"]
    assert settings.allow_fallbacks is False
    assert settings.require_parameters is True


def test_openrouter_settings_demo_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Demo profile allows fallbacks, default temperature 0.7, and no pinned provider."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENROUTER_LLM_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")
    monkeypatch.setenv("OPENROUTER_EMBEDDING_DIMENSIONS", "1536")
    monkeypatch.setenv("GRAPH_RAG_PROFILE", "demo")
    monkeypatch.delenv("GRAPH_RAG_TEMPERATURE", raising=False)
    monkeypatch.delenv("GRAPH_RAG_SEED", raising=False)
    monkeypatch.delenv("OPENROUTER_PROVIDER_ORDER", raising=False)
    monkeypatch.delenv("OPENROUTER_ALLOW_FALLBACKS", raising=False)

    settings = OpenRouterSettings.from_env()

    assert settings.profile == "demo"
    assert settings.temperature == 0.7
    assert settings.seed is None
    assert settings.provider_order is None
    assert settings.allow_fallbacks is True


def test_openrouter_settings_explicit_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit environment variables override profile defaults."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENROUTER_LLM_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")
    monkeypatch.setenv("OPENROUTER_EMBEDDING_DIMENSIONS", "1536")
    monkeypatch.setenv("GRAPH_RAG_PROFILE", "experiment")
    monkeypatch.setenv("GRAPH_RAG_TEMPERATURE", "0.2")
    monkeypatch.setenv("GRAPH_RAG_SEED", "42")
    monkeypatch.setenv("OPENROUTER_PROVIDER_ORDER", "anthropic, openai")
    monkeypatch.setenv("OPENROUTER_ALLOW_FALLBACKS", "true")

    settings = OpenRouterSettings.from_env()

    assert settings.temperature == 0.2
    assert settings.seed == 42
    assert settings.provider_order == ["anthropic", "openai"]
    assert settings.allow_fallbacks is True


def test_is_reasoning_model_detection() -> None:
    """Reasoning models (o1, o3, deepseek-r1) are detected; others are not."""
    assert is_reasoning_model("openai/o1-mini") is True
    assert is_reasoning_model("openai/o1-preview") is True
    assert is_reasoning_model("openai/o3-mini") is True
    assert is_reasoning_model("deepseek/deepseek-r1") is True
    assert is_reasoning_model("deepseek/deepseek-reasoner") is True
    assert is_reasoning_model("openai/gpt-4o-mini") is False
    assert is_reasoning_model("openai/gpt-4o") is False
    assert is_reasoning_model("anthropic/claude-3.5-sonnet") is False
    assert is_reasoning_model("qwen3:4b") is False

