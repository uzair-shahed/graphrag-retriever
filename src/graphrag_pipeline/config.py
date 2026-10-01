"""Environment-based configuration for the local application runtime.

Values are read from process environment variables, optionally populated from
a local `.env` file (see `.env.example`). This module owns configuration
parsing only; it does not open any connections.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

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

        return cls(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            llm_model=llm_model,
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
            llm_max_tokens=_get_positive_int_env("OPENROUTER_LLM_MAX_TOKENS", 4096),
            require_parameters=_get_bool_env("OPENROUTER_REQUIRE_PARAMETERS", True),
            allow_fallbacks=_get_bool_env("OPENROUTER_ALLOW_FALLBACKS", True),
        )
