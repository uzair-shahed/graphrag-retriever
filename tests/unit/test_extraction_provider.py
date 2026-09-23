"""Unit tests for the Contract 2 extraction provider boundary.

Last modified: 2026-09-23

Purpose of this file:
    Verify that a provider accepts validated extraction requests, returns raw
    structured payloads without paid model calls, and does not own result
    validation or retry behavior.

Files that use this file:
    Pytest discovers this module. It imports the provider boundary from
    ``src/graphrag_pipeline/extraction/provider.py`` and loads the request from
    ``tests/fixtures/musique/extraction_chunk.json``.

Caveats:
    These tests do not implement or verify the extraction agent, retries,
    dead-letter persistence, external LLM providers, or LangGraph wiring.
"""

import json
from pathlib import Path

from graphrag_pipeline.extraction.provider import (
    DeterministicMockProvider,
    ExtractionProvider,
)
from graphrag_pipeline.extraction.schemas import ExtractionRequest


FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "musique" / "extraction_chunk.json"


def load_request() -> ExtractionRequest:
    """Load the validated request used by provider tests.

    Inputs:
        None. The request is read from ``FIXTURE_PATH``.
    Outputs:
        A validated ``ExtractionRequest``.
    """
    return ExtractionRequest.model_validate(json.loads(FIXTURE_PATH.read_text()))


def test_mock_provider_implements_provider_protocol() -> None:
    """Ensure the deterministic mock satisfies the provider boundary.

    Inputs:
        None.
    Outputs:
        No return value; passes when the mock is recognized as an
        ``ExtractionProvider`` implementation.
    """
    assert isinstance(DeterministicMockProvider(), ExtractionProvider)


def test_mock_provider_returns_identity_preserving_empty_response() -> None:
    """Return a usable offline response without exposing gold labels.

    Inputs:
        None.
    Outputs:
        No return value; passes when the response preserves request identity,
        provenance, and text while containing empty extraction collections.
    """
    request = load_request()
    provider = DeterministicMockProvider()

    response = provider.extract(request)

    assert response["chunk_id"] == request.chunk_id
    assert response["record_id"] == request.record_id
    assert response["chunk_text"] == request.text
    assert response["provenance"] == request.provenance.model_dump()
    assert response["mentions"] == []
    assert response["triples"] == []
    assert "is_supporting" not in response
    assert provider.requests == [request]


def test_mock_provider_uses_factory_and_returns_independent_payloads() -> None:
    """Support deterministic fixture-specific responses without shared state.

    Inputs:
        None.
    Outputs:
        No return value; passes when factory output is copied independently for
        each provider call and request history is preserved.
    """
    request = load_request()

    def response_factory(current_request: ExtractionRequest) -> dict[str, object]:
        return {
            "chunk_id": current_request.chunk_id,
            "record_id": current_request.record_id,
            "chunk_text": current_request.text,
            "provenance": current_request.provenance.model_dump(),
            "mentions": [],
            "triples": [],
        }

    provider = DeterministicMockProvider(response_factory)
    first_response = provider.extract(request)
    first_response["mentions"].append({"unexpected": "mutation"})
    second_response = provider.extract(request)

    assert second_response["mentions"] == []
    assert len(provider.requests) == 2


def test_mock_provider_rejects_non_mapping_factory_output() -> None:
    """Expose a provider implementation error instead of hiding it.

    Inputs:
        None.
    Outputs:
        No return value; passes when a factory returning a non-dict raises
        ``TypeError`` before the extraction agent receives the response.
    """
    request = load_request()
    provider = DeterministicMockProvider(lambda _: "invalid")  # type: ignore[arg-type]

    try:
        provider.extract(request)
    except TypeError as error:
        assert str(error) == "response_factory must return a dict"
    else:
        raise AssertionError("provider accepted a non-dict response")