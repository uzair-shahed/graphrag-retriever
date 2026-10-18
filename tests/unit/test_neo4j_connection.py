"""Unit tests for `Neo4jConnection` retry/timeout behavior using a fake driver.

These tests do not require a running Neo4j instance; they replace the driver
factory with an in-memory fake to verify bounded timeouts and bounded retries.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest
from neo4j.exceptions import ClientError, TransientError

from graphrag_pipeline.config import Neo4jSettings
from graphrag_pipeline.storage import neo4j_connection as neo4j_connection_module
from graphrag_pipeline.storage.neo4j_connection import Neo4jConnection


class _FakeRecord:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def data(self) -> dict[str, Any]:
        return self._data


class _FakeTransaction:
    def __init__(self, on_run) -> None:
        self._on_run = on_run
        self.timeout: Optional[float] = None
        self.committed = False
        self.rolled_back = False

    def run(self, query: str, parameters: dict[str, Any]):
        return self._on_run(query, parameters)

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True


class _FakeSession:
    def __init__(self, driver: "_FakeDriver") -> None:
        self._driver = driver

    def __enter__(self) -> "_FakeSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def begin_transaction(self, timeout: float) -> _FakeTransaction:
        transaction = _FakeTransaction(self._driver.next_result)
        transaction.timeout = timeout
        self._driver.transactions.append(transaction)
        return transaction


class _FakeDriver:
    def __init__(self) -> None:
        self.transactions: list[_FakeTransaction] = []
        self.session_calls: list[dict[str, Any]] = []
        self.closed = False
        self._responses: list[Any] = []

    def queue_response(self, response: Any) -> None:
        self._responses.append(response)

    def next_result(self, query: str, parameters: dict[str, Any]):
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return [_FakeRecord(row) for row in response]

    def session(self, database: str, default_access_mode: str) -> _FakeSession:
        self.session_calls.append({"database": database, "default_access_mode": default_access_mode})
        return _FakeSession(self)

    def verify_connectivity(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def _settings(**overrides: Any) -> Neo4jSettings:
    defaults = dict(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="test-password",
        database="neo4j",
        connection_timeout_seconds=5.0,
        max_transaction_retry_time_seconds=15.0,
        transaction_timeout_seconds=10.0,
    )
    defaults.update(overrides)
    return Neo4jSettings(**defaults)


@pytest.fixture()
def fake_driver(monkeypatch: pytest.MonkeyPatch) -> _FakeDriver:
    driver = _FakeDriver()
    monkeypatch.setattr(
        neo4j_connection_module.GraphDatabase, "driver", lambda *args, **kwargs: driver
    )
    # Keep retry tests fast; backoff timing is not the behavior under test.
    monkeypatch.setattr(neo4j_connection_module.time, "sleep", lambda _seconds: None)
    return driver


def test_run_write_commits_and_bounds_transaction_timeout(fake_driver: _FakeDriver) -> None:
    fake_driver.queue_response([{"id": "entity-1"}])
    connection = Neo4jConnection(_settings(transaction_timeout_seconds=2.5))

    records = connection.run_write("MERGE (e:Entity {id: $id}) RETURN e.id AS id", {"id": "entity-1"})

    assert records == [{"id": "entity-1"}]
    assert fake_driver.transactions[0].timeout == 2.5
    assert fake_driver.transactions[0].committed is True


def test_run_read_uses_read_access_mode(fake_driver: _FakeDriver) -> None:
    fake_driver.queue_response([])
    connection = Neo4jConnection(_settings())

    connection.run_read("MATCH (e:Entity) RETURN e")

    assert fake_driver.session_calls[0]["default_access_mode"] == "READ"


def test_transient_errors_are_retried_up_to_max_retries(fake_driver: _FakeDriver) -> None:
    fake_driver.queue_response(TransientError("temporary conflict"))
    fake_driver.queue_response(TransientError("temporary conflict"))
    fake_driver.queue_response([{"ok": True}])
    connection = Neo4jConnection(_settings(), max_retries=3)

    records = connection.run_write("RETURN 1")

    assert records == [{"ok": True}]
    assert len(fake_driver.transactions) == 3
    assert fake_driver.transactions[0].rolled_back is True
    assert fake_driver.transactions[1].rolled_back is True
    assert fake_driver.transactions[2].committed is True


def test_retry_is_bounded_and_reraises_after_max_retries(fake_driver: _FakeDriver) -> None:
    for _ in range(5):
        fake_driver.queue_response(TransientError("always fails"))
    connection = Neo4jConnection(_settings(), max_retries=2)

    with pytest.raises(TransientError):
        connection.run_write("RETURN 1")

    # initial attempt + max_retries retries = 3 total attempts, never more
    assert len(fake_driver.transactions) == 3


def test_non_retryable_errors_are_not_retried(fake_driver: _FakeDriver) -> None:
    fake_driver.queue_response(ClientError("bad query"))
    connection = Neo4jConnection(_settings(), max_retries=3)

    with pytest.raises(ClientError):
        connection.run_write("THIS IS NOT CYPHER")

    assert len(fake_driver.transactions) == 1
    assert fake_driver.transactions[0].rolled_back is True


def test_multiple_queries_share_one_atomic_transaction(fake_driver):
    fake_driver.queue_response([{"first": 1}])
    fake_driver.queue_response([{"second": 2}])
    connection = Neo4jConnection(_settings())

    def operation(tx):
        list(tx.run("FIRST", {}))
        return [r.data() for r in tx.run("SECOND", {})]

    assert connection.run_transaction(operation) == [{"second": 2}]
    assert len(fake_driver.transactions) == 1
    assert fake_driver.transactions[0].committed


def test_borrowed_driver_is_not_closed_by_writer_connection(fake_driver):
    connection = Neo4jConnection(_settings(), driver=fake_driver)
    connection.close()
    assert not fake_driver.closed


def test_connect_reuses_existing_driver(fake_driver: _FakeDriver) -> None:
    connection = Neo4jConnection(_settings())

    first = connection.connect()
    second = connection.connect()

    assert first is second is fake_driver


def test_close_releases_the_driver(fake_driver: _FakeDriver) -> None:
    connection = Neo4jConnection(_settings())
    connection.connect()

    connection.close()

    assert fake_driver.closed is True


def test_context_manager_closes_on_exit(fake_driver: _FakeDriver) -> None:
    with Neo4jConnection(_settings()) as connection:
        connection.connect()

    assert fake_driver.closed is True
