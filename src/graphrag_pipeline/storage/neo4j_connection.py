"""Minimal, bounded Neo4j connection wrapper.

Scope (Week 4 core task): the simplest working local Neo4j connection with
a bounded connection timeout, a bounded per-transaction timeout, and bounded
retry behavior for transient failures. This module owns connection lifecycle
and transaction execution only; query/schema logic belongs to other modules.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional, TypeVar

from neo4j import READ_ACCESS, WRITE_ACCESS, Driver, GraphDatabase, Session
from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError

from graphrag_pipeline.config import Neo4jSettings

logger = logging.getLogger(__name__)
T = TypeVar("T")

# Failures worth retrying: the server/cluster is temporarily unavailable or
# the transaction hit a transient conflict. Everything else (auth, schema,
# syntax errors) is not retried because retrying cannot fix it.
_RETRYABLE_EXCEPTIONS = (ServiceUnavailable, SessionExpired, TransientError)


class Neo4jConnection:
    """Wraps a Neo4j `Driver` with bounded timeouts and bounded retries."""

    def __init__(self, settings: Neo4jSettings, max_retries: int = 3, *, driver: Driver | None = None) -> None:
        self._settings = settings
        self._max_retries = max_retries
        self._driver: Optional[Driver] = driver
        self._owns_driver = driver is None

    def connect(self) -> Driver:
        """Lazily create the driver. Does not open a network connection yet."""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self._settings.uri,
                auth=(self._settings.username, self._settings.password),
                connection_timeout=self._settings.connection_timeout_seconds,
            )
        return self._driver

    def verify_connectivity(self) -> None:
        """Eagerly confirm the driver can reach the local Neo4j instance."""
        self.connect().verify_connectivity()

    def close(self) -> None:
        if self._driver is not None and self._owns_driver:
            self._driver.close()
            self._driver = None

    def __enter__(self) -> "Neo4jConnection":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def run_write(self, query: str, parameters: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        """Run one write query in a single bounded, retried transaction."""
        return self._run_with_retry(query, parameters, access_mode=WRITE_ACCESS)

    def run_read(self, query: str, parameters: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        """Run one read query in a single bounded, retried transaction."""
        return self._run_with_retry(query, parameters, access_mode=READ_ACCESS)

    def _session(self, access_mode: str) -> Session:
        return self.connect().session(database=self._settings.database, default_access_mode=access_mode)

    def _run_with_retry(
        self,
        query: str,
        parameters: Optional[dict[str, Any]],
        access_mode: str,
    ) -> list[dict[str, Any]]:
        return self.run_transaction(
            lambda tx: [record.data() for record in tx.run(query, parameters or {})],
            access_mode=access_mode,
        )

    def run_transaction(self, operation: Callable[[Any], T], *, access_mode: str = WRITE_ACCESS) -> T:
        """Run all callback queries atomically; callbacks may be replayed after transient errors.

        Keep provider calls outside this boundary. A lost commit acknowledgment
        must be reconciled by the callback's database identity guard on retry.
        """
        deadline = time.monotonic() + self._settings.max_transaction_retry_time_seconds
        attempt = 0

        while True:
            attempt += 1
            try:
                with self._session(access_mode) as session:
                    transaction = session.begin_transaction(
                        timeout=self._settings.transaction_timeout_seconds
                    )
                    try:
                        value = operation(transaction)
                        transaction.commit()
                        return value
                    except BaseException:
                        try:
                            transaction.rollback()
                        except Exception:
                            # A commit can succeed before its acknowledgment is
                            # lost. Rollback of that closed transaction must not
                            # hide the original retryable/interrupt exception.
                            logger.debug("Rollback unavailable after transaction failure", exc_info=True)
                        raise
            except _RETRYABLE_EXCEPTIONS as error:
                remaining_seconds = deadline - time.monotonic()
                if attempt > self._max_retries or remaining_seconds <= 0:
                    logger.error(
                        "Neo4j transaction failed after %d attempt(s): %s", attempt, error
                    )
                    raise
                backoff_seconds = min(2 ** (attempt - 1) * 0.5, remaining_seconds)
                logger.warning(
                    "Retrying Neo4j transaction (attempt %d/%d) after transient error: %s",
                    attempt,
                    self._max_retries,
                    error,
                )
                time.sleep(backoff_seconds)
