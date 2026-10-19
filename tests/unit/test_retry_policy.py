"""Stage/cause classification and budget tests use actual SDK exception types."""

import json

import httpx
import pytest
from neo4j.exceptions import AuthError, ClientError, ServiceUnavailable, SessionExpired, TransientError
from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.experimental.pipeline.exceptions import InvalidJSONError
from openai import APIConnectionError, APIStatusError
from pydantic import ValidationError

from graphrag_pipeline.extraction.atomic_writer import IngestionStateError
from graphrag_pipeline.ingestion.retry_policy import RetryPolicy, classify_failure, exception_chain


@pytest.mark.parametrize("kwargs", [
    {"max_attempts": 0}, {"max_attempts": True}, {"max_attempts": "3"},
    {"delays_seconds": (-1, 30)}, {"delays_seconds": (float("nan"), 30)},
    {"delays_seconds": (float("inf"), 30)}, {"delays_seconds": (True, 30)},
    {"delays_seconds": (5,)}, {"extra": 1},
])
def test_invalid_policy_rejected(kwargs):
    with pytest.raises(ValidationError):
        RetryPolicy(**kwargs)


def test_defaults_and_environment(monkeypatch):
    monkeypatch.delenv("INGESTION_RETRY_MAX_ATTEMPTS", raising=False)
    monkeypatch.delenv("INGESTION_RETRY_DELAYS_SECONDS", raising=False)
    assert RetryPolicy.from_env().delays_seconds == (5, 30)
    monkeypatch.setenv("INGESTION_RETRY_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("INGESTION_RETRY_DELAYS_SECONDS", "[1.5]")
    assert RetryPolicy.from_env().delay_after(1) == 1.5
    with pytest.raises(ValueError):
        RetryPolicy.from_env().delay_after(2)
    monkeypatch.setenv("INGESTION_RETRY_DELAYS_SECONDS", "not json")
    with pytest.raises(ValueError):
        RetryPolicy.from_env()


@pytest.mark.parametrize("error,category", [
    (TimeoutError(), "timeout"), (httpx.ReadTimeout("timeout"), "timeout"),
    (ConnectionError(), "connectivity_or_transient_database"),
    (httpx.ConnectError("connection"), "connectivity_or_transient_database"),
    (ServiceUnavailable(), "connectivity_or_transient_database"),
    (SessionExpired(), "connectivity_or_transient_database"),
    (TransientError(), "connectivity_or_transient_database"),
    (APIConnectionError(request=httpx.Request("POST", "http://unused")), "connectivity_or_transient_database"),
])
def test_transient_errors_survive_sdk_wrapping(error, category):
    wrapper = LLMGenerationError(error)
    decision = classify_failure(wrapper, "embedding")
    assert decision.retryable and decision.category == category


@pytest.mark.parametrize("status,retryable,category", [
    (401, False, "authentication"), (403, False, "authentication"),
    (400, False, "invalid_request"), (404, False, "invalid_request"),
    (408, True, "temporary_service"), (429, True, "rate_limit"),
    (500, True, "temporary_service"), (503, True, "temporary_service"),
])
def test_http_error_categories(status, retryable, category):
    error = APIStatusError("provider error", response=httpx.Response(status, request=httpx.Request("POST", "http://unused")), body=None)
    wrapper = RuntimeError("pipeline")
    wrapper.__cause__ = error
    decision = classify_failure(wrapper, "extraction")
    assert (decision.retryable, decision.category) == (retryable, category)


@pytest.mark.parametrize("error", [json.JSONDecodeError("bad", "{", 1), InvalidJSONError("bad")])
def test_malformed_generation_is_retryable_only_at_output_boundary(error):
    wrapper = LLMGenerationError("invalid output")
    wrapper.__cause__ = error
    assert classify_failure(wrapper, "extraction").retryable
    assert not classify_failure(wrapper, "input").retryable
    assert not classify_failure(wrapper, "embedding").retryable


def test_schema_and_graph_validation_are_stage_aware():
    with pytest.raises(ValidationError) as captured:
        RetryPolicy(max_attempts=0)
    assert classify_failure(captured.value, "extraction").category == "malformed_model_output"
    assert classify_failure(captured.value, "configuration").category == "configuration"
    assert classify_failure(ValueError("bad endpoints"), "graph_validation").retryable
    assert not classify_failure(ValueError("unknown bug"), "write").retryable


@pytest.mark.parametrize("error,category", [
    (AuthError(), "authentication"), (ClientError(), "database_client"),
    (IngestionStateError("partial graph"), "unsafe_graph_state"),
    (RuntimeError("unknown"), "unknown"),
])
def test_terminal_errors_override_nested_transient_causes(error, category):
    error.__cause__ = TimeoutError()
    decision = classify_failure(error, "write")
    if category == "unknown":
        # A known transient cause inside a generic pipeline wrapper is retryable.
        assert decision.retryable
        assert not classify_failure(RuntimeError("unknown"), "write").retryable
    else:
        assert not decision.retryable and decision.category == category


def test_exception_chain_cycle_and_stage_validation():
    error = RuntimeError("cycle")
    error.__cause__ = error
    assert exception_chain(error) == [error]
    with pytest.raises(ValueError, match="unknown ingestion failure stage"):
        classify_failure(error, "invalid")


def test_graphrag_rate_limit_and_retryable_error_are_supported():
    from neo4j_graphrag.exceptions import RateLimitError, RetryableError

    assert classify_failure(RateLimitError("rate limited"), "extraction").category == "rate_limit"
    assert classify_failure(RetryableError("temporary"), "extraction").retryable


def test_configuration_and_unsafe_state_override_retryable_causes():
    error = IngestionStateError("inventory mismatch")
    error.__cause__ = httpx.ConnectError("connection")
    assert not classify_failure(error, "reconcile").retryable
    assert not classify_failure(TimeoutError(), "configuration").retryable
    error = RuntimeError("suppressed context")
    error.__context__ = TimeoutError()
    error.__suppress_context__ = True
    assert not classify_failure(error, "write").retryable
