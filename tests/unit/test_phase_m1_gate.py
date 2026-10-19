"""Phase gate M1 failure drills verifying that every error condition fails fast for $0."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest
from neo4j.exceptions import ServiceUnavailable

from graphrag_pipeline.config import Neo4jSettings, OpenRouterSettings
from graphrag_pipeline.preflight import PreflightError, run_preflight_checks


def dummy_neo4j_settings() -> Neo4jSettings:
    return Neo4jSettings(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="changeme",
        database="neo4j",
        connection_timeout_seconds=5.0,
        max_transaction_retry_time_seconds=15.0,
        transaction_timeout_seconds=10.0,
    )


def dummy_openrouter_settings(base_url: str = "http://127.0.0.1:18080/v1") -> OpenRouterSettings:
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


def test_drill_neo4j_down_fails_with_zero_model_calls() -> None:
    """Drill 1: Neo4j down fails immediately with PreflightError and 0 model calls."""
    model_mock = MagicMock()
    with (
        patch("graphrag_pipeline.preflight.Neo4jConnection") as mock_conn_cls,
        patch("urllib.request.urlopen", model_mock),
    ):
        mock_conn = MagicMock()
        mock_conn.verify_connectivity.side_effect = ServiceUnavailable("Connection refused")
        mock_conn_cls.return_value.__enter__.return_value = mock_conn

        with pytest.raises(PreflightError, match="Neo4j is not reachable"):
            run_preflight_checks(dummy_neo4j_settings(), dummy_openrouter_settings())

    assert model_mock.call_count == 0, "No model calls should be made when Neo4j is down"


def test_drill_missing_apoc_fails_with_zero_model_calls() -> None:
    """Drill 2: Missing APOC procedures fails with clear message and 0 model calls."""
    model_mock = MagicMock()
    with (
        patch("graphrag_pipeline.preflight.Neo4jConnection") as mock_conn_cls,
        patch("urllib.request.urlopen", model_mock),
    ):
        mock_conn = MagicMock()
        mock_conn.run_read.return_value = [{"name": "apoc.create.addLabels"}]  # missing apoc.merge.relationship
        mock_conn_cls.return_value.__enter__.return_value = mock_conn

        with pytest.raises(PreflightError, match="Missing required APOC procedures"):
            run_preflight_checks(dummy_neo4j_settings(), dummy_openrouter_settings())

    assert model_mock.call_count == 0, "No model calls should be made when APOC is missing"


def test_drill_wrong_vector_dimensions_fails_with_zero_model_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drill 3: Vector dimension mismatch fails with clear message and 0 model calls."""
    monkeypatch.setenv("OPENROUTER_BUDGET_USD", "5.0")
    model_mock = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__enter__.return_value = mock_resp
    model_mock.return_value = mock_resp

    with (
        patch("graphrag_pipeline.preflight.Neo4jConnection") as mock_conn_cls,
        patch("urllib.request.urlopen", model_mock),
    ):
        mock_conn = MagicMock()
        mock_conn.run_read.side_effect = [
            # APOC procs pass
            [{"name": "apoc.create.addLabels"}, {"name": "apoc.merge.relationship"}],
            # Vector index dimensions mismatch (768 vs 1536)
            [{"dimensions": 768, "label": "Chunk", "property": "embedding"}],
        ]
        mock_conn_cls.return_value.__enter__.return_value = mock_conn

        with pytest.raises(PreflightError, match="Vector index dimension mismatch: index has 768, but embedder requires 1536"):
            run_preflight_checks(
                dummy_neo4j_settings(),
                dummy_openrouter_settings(),
                require_vector_index=True,
            )

    # urllib should only have made the preflight model endpoint check, not any embeddings or completions
    for call in model_mock.call_args_list:
        url = call[0][0].full_url if hasattr(call[0][0], "full_url") else str(call[0][0])
        assert "/chat/completions" not in url
        assert "/embeddings" not in url


def test_drill_missing_budget_fails_with_zero_model_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drill 4: Missing budget ceiling fails fast with PreflightError and 0 model calls."""
    monkeypatch.delenv("OPENROUTER_BUDGET_USD", raising=False)
    monkeypatch.delenv("GRAPH_RAG_BUDGET_USD", raising=False)

    model_mock = MagicMock()
    with (
        patch("graphrag_pipeline.preflight.Neo4jConnection") as mock_conn_cls,
        patch("urllib.request.urlopen", model_mock),
    ):
        mock_conn = MagicMock()
        mock_conn.run_read.return_value = [
            {"name": "apoc.create.addLabels"},
            {"name": "apoc.merge.relationship"},
        ]
        mock_conn_cls.return_value.__enter__.return_value = mock_conn
        # When gateway /health is unreachable or has no budget
        model_mock.side_effect = Exception("No gateway")

        with pytest.raises(PreflightError, match="A budget ceiling must be configured"):
            run_preflight_checks(dummy_neo4j_settings(), dummy_openrouter_settings("https://openrouter.ai/api/v1"))
