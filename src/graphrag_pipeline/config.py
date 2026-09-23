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
