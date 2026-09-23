"""Integration smoke test against a real local Neo4j instance.

Skipped automatically when Neo4j is not reachable or `.env`/environment
variables are not configured, so the unit test suite never requires a
running database.
"""

from __future__ import annotations

import pytest
from neo4j.exceptions import ServiceUnavailable

from graphrag_pipeline.config import Neo4jSettings
from graphrag_pipeline.storage.neo4j_connection import Neo4jConnection


def _load_settings_or_skip() -> Neo4jSettings:
    try:
        return Neo4jSettings.from_env()
    except ValueError:
        pytest.skip("NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD not configured; skipping integration test.")


def test_local_neo4j_connection_smoke_test() -> None:
    settings = _load_settings_or_skip()

    with Neo4jConnection(settings) as connection:
        try:
            connection.verify_connectivity()
        except ServiceUnavailable:
            pytest.skip("Local Neo4j is not reachable; skipping integration test.")

        connection.run_write(
            "MERGE (n:ConnectionSmokeTest {id: $id}) SET n.checked_at = timestamp()",
            {"id": "connection-smoke-test"},
        )
        records = connection.run_read(
            "MATCH (n:ConnectionSmokeTest {id: $id}) RETURN n.id AS id",
            {"id": "connection-smoke-test"},
        )
        assert records == [{"id": "connection-smoke-test"}]

        connection.run_write("MATCH (n:ConnectionSmokeTest {id: $id}) DELETE n", {"id": "connection-smoke-test"})
