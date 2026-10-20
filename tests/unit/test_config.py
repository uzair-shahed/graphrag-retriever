"""Unit tests for `Neo4jSettings` env parsing. No network access required."""

from __future__ import annotations

import pytest

from graphrag_pipeline.config import Neo4jSettings, OllamaSettings, model_settings_from_env


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


def test_ollama_request_limits_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_REQUEST_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("OLLAMA_LLM_MAX_TOKENS", "1024")
    settings = OllamaSettings.from_env()
    assert settings.request_timeout_seconds == 45
    assert settings.llm_max_tokens == 1024


@pytest.mark.parametrize("name", ["OLLAMA_REQUEST_TIMEOUT_SECONDS", "OLLAMA_LLM_MAX_TOKENS"])
def test_ollama_rejects_unbounded_limits(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setenv(name, "0")
    with pytest.raises(ValueError, match=name):
        OllamaSettings.from_env()


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
