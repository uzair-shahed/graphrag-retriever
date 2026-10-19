"""Unit tests for preflight checks verifying database and model configuration before spend."""

from __future__ import annotations

import json
import urllib.error
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from neo4j.exceptions import ServiceUnavailable

from graphrag_pipeline.config import Neo4jSettings, OllamaSettings, OpenRouterSettings
from graphrag_pipeline.preflight import (
    PreflightError,
    check_budget_configured,
    check_model_endpoint,
    check_neo4j_connectivity_and_apoc,
    check_vector_index,
    run_preflight_checks,
)


def sample_neo4j_settings() -> Neo4jSettings:
    return Neo4jSettings(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="changeme",
        database="neo4j",
        connection_timeout_seconds=5.0,
        max_transaction_retry_time_seconds=15.0,
        transaction_timeout_seconds=10.0,
    )


def sample_openrouter_settings(base_url: str = "http://127.0.0.1:18080/v1") -> OpenRouterSettings:
    return OpenRouterSettings(
        api_key="test-key",
        base_url=base_url,
        llm_model="openai/gpt-4o-mini",
        embedding_model="openai/text-embedding-3-small",
        embedding_dimensions=1536,
        llm_max_tokens=4096,
        require_parameters=True,
        allow_fallbacks=True,
    )


def test_neo4j_connectivity_and_apoc_success() -> None:
    mock_conn = MagicMock()
    mock_conn.run_read.return_value = [
        {"name": "apoc.create.addLabels"},
        {"name": "apoc.merge.relationship"},
    ]
    with patch("graphrag_pipeline.preflight.Neo4jConnection") as mock_cls:
        mock_cls.return_value.__enter__.return_value = mock_conn
        check_neo4j_connectivity_and_apoc(sample_neo4j_settings())


def test_neo4j_connectivity_failure_raises() -> None:
    mock_conn = MagicMock()
    mock_conn.verify_connectivity.side_effect = ServiceUnavailable("Cannot connect")
    with patch("graphrag_pipeline.preflight.Neo4jConnection") as mock_cls:
        mock_cls.return_value.__enter__.return_value = mock_conn
        with pytest.raises(PreflightError, match="Neo4j is not reachable"):
            check_neo4j_connectivity_and_apoc(sample_neo4j_settings())


def test_neo4j_missing_apoc_raises() -> None:
    mock_conn = MagicMock()
    mock_conn.run_read.return_value = [{"name": "apoc.create.addLabels"}]
    with patch("graphrag_pipeline.preflight.Neo4jConnection") as mock_cls:
        mock_cls.return_value.__enter__.return_value = mock_conn
        with pytest.raises(PreflightError, match="Missing required APOC procedures.*apoc.merge.relationship"):
            check_neo4j_connectivity_and_apoc(sample_neo4j_settings())


def test_vector_index_success() -> None:
    mock_conn = MagicMock()
    mock_conn.run_read.side_effect = [
        [{"dimensions": 1536, "label": "Chunk", "property": "embedding"}],
        [{"state": "ONLINE"}],
    ]
    with patch("graphrag_pipeline.preflight.Neo4jConnection") as mock_cls:
        mock_cls.return_value.__enter__.return_value = mock_conn
        check_vector_index(sample_neo4j_settings(), expected_dim=1536)


def test_vector_index_missing_raises() -> None:
    mock_conn = MagicMock()
    mock_conn.run_read.return_value = []
    with patch("graphrag_pipeline.preflight.Neo4jConnection") as mock_cls:
        mock_cls.return_value.__enter__.return_value = mock_conn
        with pytest.raises(PreflightError, match="Vector index 'chunk_embedding' does not exist"):
            check_vector_index(sample_neo4j_settings(), expected_dim=1536)


def test_vector_index_dimension_mismatch_raises() -> None:
    mock_conn = MagicMock()
    mock_conn.run_read.return_value = [{"dimensions": 768, "label": "Chunk", "property": "embedding"}]
    with patch("graphrag_pipeline.preflight.Neo4jConnection") as mock_cls:
        mock_cls.return_value.__enter__.return_value = mock_conn
        with pytest.raises(PreflightError, match="Vector index dimension mismatch"):
            check_vector_index(sample_neo4j_settings(), expected_dim=1536)


def test_vector_index_not_online_raises() -> None:
    mock_conn = MagicMock()
    mock_conn.run_read.side_effect = [
        [{"dimensions": 1536, "label": "Chunk", "property": "embedding"}],
        [{"state": "POPULATING"}],
    ]
    with patch("graphrag_pipeline.preflight.Neo4jConnection") as mock_cls:
        mock_cls.return_value.__enter__.return_value = mock_conn
        with pytest.raises(PreflightError, match="is not ONLINE: state is POPULATING"):
            check_vector_index(sample_neo4j_settings(), expected_dim=1536)


def test_budget_configured_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_BUDGET_USD", "10.0")
    check_budget_configured(sample_openrouter_settings())


def test_budget_configured_from_gateway_health(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_BUDGET_USD", raising=False)
    monkeypatch.delenv("GRAPH_RAG_BUDGET_USD", raising=False)

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps({"budget_usd": 5.0}).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        check_budget_configured(sample_openrouter_settings("http://127.0.0.1:18080/v1"))


def test_budget_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_BUDGET_USD", raising=False)
    monkeypatch.delenv("GRAPH_RAG_BUDGET_USD", raising=False)

    with patch("urllib.request.urlopen", side_effect=Exception("Gateway unreachable")):
        with pytest.raises(PreflightError, match="A budget ceiling must be configured"):
            check_budget_configured(sample_openrouter_settings("https://openrouter.ai/api/v1"))


def test_budget_invalid_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_BUDGET_USD", "-5.0")
    with pytest.raises(PreflightError, match="Budget ceiling must be positive"):
        check_budget_configured(sample_openrouter_settings())


def test_model_endpoint_openrouter_success() -> None:
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__enter__.return_value = mock_resp
    with patch("urllib.request.urlopen", return_value=mock_resp):
        check_model_endpoint(sample_openrouter_settings())


def test_model_endpoint_auth_failure() -> None:
    http_error = urllib.error.HTTPError(
        url="http://127.0.0.1:18080/v1/models",
        code=401,
        msg="Unauthorized",
        hdrs={},  # type: ignore[arg-type]
        fp=None,
    )
    with patch("urllib.request.urlopen", side_effect=http_error):
        with pytest.raises(PreflightError, match="Model endpoint authentication failed: HTTP 401"):
            check_model_endpoint(sample_openrouter_settings())


def test_model_endpoint_unreachable() -> None:
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
        with pytest.raises(PreflightError, match="Model endpoint is unreachable"):
            check_model_endpoint(sample_openrouter_settings())


def test_run_preflight_checks_full(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_BUDGET_USD", "5.0")
    mock_conn = MagicMock()
    mock_conn.run_read.side_effect = [
        # APOC procs
        [{"name": "apoc.create.addLabels"}, {"name": "apoc.merge.relationship"}],
        # Vector index
        [{"dimensions": 1536, "label": "Chunk", "property": "embedding"}],
        # Vector index state
        [{"state": "ONLINE"}],
    ]
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__enter__.return_value = mock_resp

    with (
        patch("graphrag_pipeline.preflight.Neo4jConnection") as mock_cls,
        patch("urllib.request.urlopen", return_value=mock_resp),
    ):
        mock_cls.return_value.__enter__.return_value = mock_conn
        run_preflight_checks(
            sample_neo4j_settings(),
            sample_openrouter_settings(),
            require_vector_index=True,
        )
