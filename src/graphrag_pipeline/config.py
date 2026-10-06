"""Environment-based configuration for the local application runtime.

Values are read from process environment variables, optionally populated from
a local `.env` file (see `.env.example`). This module owns configuration
parsing only; it does not open any connections.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _get_float_env(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        return float(raw_value)
    except ValueError as error:
        raise ValueError(f"Environment variable {name} must be a number, got {raw_value!r}") from error


def _get_bool_env(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Environment variable {name} must be a boolean, got {raw_value!r}")


def _get_positive_int_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw_value!r}") from error
    if value <= 0:
        raise ValueError(f"Environment variable {name} must be a positive integer.")
    return value


def _get_nonnegative_int_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw_value!r}") from error
    if value < 0:
        raise ValueError(f"Environment variable {name} must be non-negative.")
    return value


def dataset_paths_from_env() -> list[Path]:
    """Read the configured comma-separated MuSiQue JSONL paths.

    Returns:
        Dataset paths in the order configured by ``GRAPH_RAG_DATASET_PATHS``.

    Raises:
        ValueError: If the setting is missing or contains an empty path.
    """
    raw_value = os.environ.get("GRAPH_RAG_DATASET_PATHS", "")
    paths = [Path(item.strip()) for item in raw_value.split(",") if item.strip()]
    if not paths:
        raise ValueError(
            "GRAPH_RAG_DATASET_PATHS must contain one or more comma-separated JSONL paths."
        )
    return paths


def _get_chunk_settings() -> tuple[int, int]:
    chunk_size = _get_positive_int_env("GRAPH_RAG_CHUNK_SIZE", 10000)
    chunk_overlap = _get_nonnegative_int_env("GRAPH_RAG_CHUNK_OVERLAP", 0)
    if chunk_overlap >= chunk_size:
        raise ValueError("GRAPH_RAG_CHUNK_OVERLAP must be less than GRAPH_RAG_CHUNK_SIZE.")
    return chunk_size, chunk_overlap


@dataclass(frozen=True)
class Neo4jSettings:
    """Bounded connection and transaction settings for the local Neo4j instance."""

    uri: str
    username: str
    password: str
    database: str
    connection_timeout_seconds: float
    max_transaction_retry_time_seconds: float
    transaction_timeout_seconds: float

    @classmethod
    def from_env(cls) -> "Neo4jSettings":
        uri = os.environ.get("NEO4J_URI")
        username = os.environ.get("NEO4J_USERNAME")
        password = os.environ.get("NEO4J_PASSWORD")
        if not uri or not username or not password:
            raise ValueError(
                "NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD must be set "
                "(copy .env.example to .env and fill in local values)."
            )

        return cls(
            uri=uri,
            username=username,
            password=password,
            database=os.environ.get("NEO4J_DATABASE", "neo4j"),
            connection_timeout_seconds=_get_float_env("NEO4J_CONNECTION_TIMEOUT_SECONDS", 5.0),
            max_transaction_retry_time_seconds=_get_float_env(
                "NEO4J_MAX_TRANSACTION_RETRY_TIME_SECONDS", 15.0
            ),
            transaction_timeout_seconds=_get_float_env("NEO4J_TRANSACTION_TIMEOUT_SECONDS", 10.0),
        )


@dataclass(frozen=True)
class OpenRouterSettings:
    """OpenRouter settings for the OpenAI-compatible GraphRAG providers."""

    api_key: str
    base_url: str
    llm_model: str
    embedding_model: str
    embedding_dimensions: int
    llm_max_tokens: int
    require_parameters: bool
    allow_fallbacks: bool
    chunk_size: int = 10000
    chunk_overlap: int = 0

    @classmethod
    def from_env(cls) -> "OpenRouterSettings":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        base_url = os.environ.get("OPENROUTER_BASE_URL")
        llm_model = os.environ.get("OPENROUTER_LLM_MODEL")
        embedding_model = os.environ.get("OPENROUTER_EMBEDDING_MODEL")
        if not api_key or not base_url or not llm_model or not embedding_model:
            raise ValueError(
                "OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_LLM_MODEL, "
                "and OPENROUTER_EMBEDDING_MODEL must be set."
            )

        raw_dimensions = os.environ.get("OPENROUTER_EMBEDDING_DIMENSIONS", "")
        try:
            embedding_dimensions = int(raw_dimensions)
        except ValueError as error:
            raise ValueError(
                "OPENROUTER_EMBEDDING_DIMENSIONS must be a positive integer."
            ) from error
        if embedding_dimensions <= 0:
            raise ValueError("OPENROUTER_EMBEDDING_DIMENSIONS must be a positive integer.")
        chunk_size, chunk_overlap = _get_chunk_settings()

        return cls(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            llm_model=llm_model,
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
            llm_max_tokens=_get_positive_int_env("OPENROUTER_LLM_MAX_TOKENS", 4096),
            require_parameters=_get_bool_env("OPENROUTER_REQUIRE_PARAMETERS", True),
            allow_fallbacks=_get_bool_env("OPENROUTER_ALLOW_FALLBACKS", True),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


@dataclass(frozen=True)
class OllamaSettings:
    """Local Ollama settings for the GraphRAG LLM and embedding providers."""

    host: str
    llm_model: str
    embedding_model: str
    chunk_size: int = 10000
    chunk_overlap: int = 0

    @classmethod
    def from_env(cls) -> "OllamaSettings":
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        llm_model = os.environ.get("OLLAMA_LLM_MODEL", "qwen3:4b")
        embedding_model = os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        if not host or not llm_model or not embedding_model:
            raise ValueError(
                "OLLAMA_HOST, OLLAMA_LLM_MODEL, and OLLAMA_EMBEDDING_MODEL "
                "must not be empty."
            )
        chunk_size, chunk_overlap = _get_chunk_settings()
        return cls(
            host=host,
            llm_model=llm_model,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


def model_settings_from_env() -> OpenRouterSettings | OllamaSettings:
    """Load model settings selected by ``GRAPH_RAG_PROVIDER``.

    Returns:
        OpenRouter or Ollama settings according to the provider switch.

    Raises:
        ValueError: If the provider name is unsupported or its settings are invalid.
    """
    provider = os.environ.get("GRAPH_RAG_PROVIDER", "openrouter").strip().lower()
    if provider == "openrouter":
        return OpenRouterSettings.from_env()
    if provider == "ollama":
        return OllamaSettings.from_env()
    raise ValueError(
        f"GRAPH_RAG_PROVIDER must be 'openrouter' or 'ollama', got {provider!r}."
    )
