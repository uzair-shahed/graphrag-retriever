from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ValidationError

from graphrag_pipeline.diagnostics import (
    exception_diagnostics,
    save_invalid_llm_response,
)


def test_exception_diagnostics_preserves_provider_error_chain_without_secrets() -> None:
    """Provider response details remain visible while credentials are redacted."""
    provider_error = RuntimeError(
        "Error code: 400 - {'message': 'model does not support json_schema', "
        "'api_key': 'sk-or-v1-secret'}"
    )
    provider_error.response = SimpleNamespace(
        text='{"error":{"message":"model does not support json_schema"}}'
    )
    outer_error = RuntimeError("Failed to generate schema from text")
    outer_error.__cause__ = provider_error

    diagnostics = exception_diagnostics(outer_error)

    assert diagnostics["error"] == "Failed to generate schema from text"
    assert diagnostics["exception_chain"][1]["type"] == "RuntimeError"
    assert "json_schema" in diagnostics["exception_chain"][1]["response_body"]["error"]["message"]
    assert "sk-or-v1-secret" not in str(diagnostics)


def test_save_invalid_llm_response_writes_validation_failure_response(tmp_path) -> None:
    """Save the exact raw response when a Pydantic validation error is chained."""

    class ExpectedGraph(BaseModel):
        nodes: list[dict]

    with pytest.raises(ValidationError) as validation_error:
        ExpectedGraph.model_validate_json('{"nodes": "not-a-list"}')

    outer_error = RuntimeError("GraphRAG extraction failed")
    outer_error.__cause__ = validation_error.value

    path = save_invalid_llm_response(
        outer_error,
        '{"nodes": "not-a-list"}',
        tmp_path / "debugging_logs",
    )

    assert path is not None
    assert path.parent == tmp_path / "debugging_logs"
    assert path.read_text(encoding="utf-8") == '{"nodes": "not-a-list"}'


def test_save_invalid_llm_response_ignores_non_validation_failure(tmp_path) -> None:
    """Do not create response logs for provider or connection failures."""
    path = save_invalid_llm_response(
        RuntimeError("provider unavailable"),
        "provider response",
        tmp_path / "debugging_logs",
    )

    assert path is None
    assert not (tmp_path / "debugging_logs").exists()


def test_bare_openrouter_token_is_fully_redacted():
    token = "sk-or-v1-abcd1234-secret"
    assert token not in str(exception_diagnostics(RuntimeError(f"rejected {token}")))
    assert "abcd1234" not in str(exception_diagnostics(RuntimeError(f"rejected {token}")))
