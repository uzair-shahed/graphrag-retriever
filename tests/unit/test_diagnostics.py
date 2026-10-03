from types import SimpleNamespace

from graphrag_pipeline.diagnostics import exception_diagnostics


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