"""Provider boundary for Contract 2 extraction responses.

Last modified: 2026-09-23

Purpose of this file:
    Define the narrow provider interface used by the extraction agent and
    provide a deterministic mock implementation for local tests without paid
    model calls.

Files that use this file:
    The future extraction agent will depend on ``ExtractionProvider``. Unit
    tests can use ``DeterministicMockProvider`` directly.

Caveats:
    This file does not parse or validate provider output, retry failed calls,
    write dead-letter records, call an external LLM, or orchestrate LangGraph.
    Those responsibilities belong to the extraction agent and later provider
    implementations.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Protocol, runtime_checkable

from graphrag_pipeline.extraction.schemas import ExtractionRequest


ResponseFactory = Callable[[ExtractionRequest], dict[str, Any]]


@runtime_checkable
class ExtractionProvider(Protocol):
    """Contract 2 provider interface consumed by the extraction agent."""

    def extract(self, request: ExtractionRequest) -> dict[str, Any]:
        """Produce one raw structured response for a validated chunk.

        Inputs:
            request: A validated Contract 2 extraction request without gold
                evaluation labels.
        Outputs:
            A raw response mapping. The extraction agent must validate it as an
            ``ExtractionResult`` before accepting it.
        """


class DeterministicMockProvider:
    """Return predictable extraction payloads for offline tests."""

    def __init__(self, response_factory: ResponseFactory | None = None) -> None:
        """Configure the deterministic response behavior.

        Inputs:
            response_factory: Optional callable that receives a request and
                returns a raw response mapping. When omitted, the provider
                returns an identity-preserving empty extraction.
        Outputs:
            None. Initializes the provider and an inspectable request history.
        """
        self._response_factory = response_factory or self._empty_response
        self.requests: list[ExtractionRequest] = []

    def extract(self, request: ExtractionRequest) -> dict[str, Any]:
        """Return one independent raw response for the supplied request.

        Inputs:
            request: A validated Contract 2 extraction request.
        Outputs:
            A deep-copied raw response mapping. Each call is recorded in
            ``requests`` and receives a fresh response object.
        """
        self.requests.append(request)
        response = self._response_factory(request)
        if not isinstance(response, dict):
            raise TypeError("response_factory must return a dict")
        return deepcopy(response)

    @staticmethod
    def _empty_response(request: ExtractionRequest) -> dict[str, Any]:
        """Build a valid empty extraction response for a request.

        Inputs:
            request: A validated Contract 2 extraction request.
        Outputs:
            A raw response with matching identity and provenance and no
            extracted mentions or triples.
        """
        return {
            "chunk_id": request.chunk_id,
            "record_id": request.record_id,
            "chunk_text": request.text,
            "provenance": request.provenance.model_dump(),
            "mentions": [],
            "triples": [],
        }