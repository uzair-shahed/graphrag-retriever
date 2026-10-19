"""Chunk-level retry decisions; independent of provider/transaction retries."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Literal

import httpx
from neo4j.exceptions import AuthError, ClientError, ServiceUnavailable, SessionExpired, TransientError
from neo4j_graphrag.experimental.pipeline.exceptions import InvalidJSONError
from neo4j_graphrag.exceptions import RateLimitError, RetryableError
from openai import APIConnectionError, APITimeoutError
from pydantic import BaseModel, ConfigDict, StrictFloat, StrictInt, ValidationError, model_validator

Stage = Literal["input", "configuration", "reconcile", "extraction", "embedding", "graph_validation", "write"]


class RetryPolicy(BaseModel):
    """One persisted cycle's budget includes its initial pipeline attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    max_attempts: StrictInt = 3
    delays_seconds: tuple[StrictInt | StrictFloat, ...] = (5, 30)

    @model_validator(mode="after")
    def valid_schedule(self):
        if self.max_attempts < 1 or len(self.delays_seconds) != self.max_attempts - 1:
            raise ValueError("provide exactly max_attempts - 1 retry delays")
        if any(not math.isfinite(delay) or delay < 0 for delay in self.delays_seconds):
            raise ValueError("retry delays must be finite and nonnegative")
        return self

    def delay_after(self, attempts: int) -> float:
        if not 1 <= attempts < self.max_attempts:
            raise ValueError("no retry delay exists for this attempt count")
        return float(self.delays_seconds[attempts - 1])

    @classmethod
    def from_env(cls):
        """Optional overrides: INGESTION_RETRY_MAX_ATTEMPTS and _DELAYS_SECONDS.

        Delays are a JSON array, e.g. '[5, 30]'. Configuration is captured when
        a cycle starts, so later environment changes cannot reset its budget.
        """
        return cls(
            max_attempts=int(os.environ.get("INGESTION_RETRY_MAX_ATTEMPTS", "3")),
            delays_seconds=tuple(json.loads(os.environ.get("INGESTION_RETRY_DELAYS_SECONDS", "[5, 30]"))),
        )


@dataclass(frozen=True)
class FailureDecision:
    category: str
    retryable: bool


def exception_chain(error: BaseException) -> list[BaseException]:
    """Include SDK wrappers that retain their cause only as an argument."""
    result, pending, seen = [], [error], set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        result.append(current)
        pending.extend(arg for arg in current.args if isinstance(arg, BaseException))
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        elif current.__context__ is not None and not current.__suppress_context__:
            pending.append(current.__context__)
    return result


def classify_failure(error: BaseException, stage: Stage) -> FailureDecision:
    """Use typed causes and stage, never message substring guesses.

    Input/configuration and unsafe graph state take precedence over nested
    transient errors. Unknown failures are retained for explicit review.
    """
    from graphrag_pipeline.extraction.atomic_writer import IngestionStateError

    if stage not in {"input", "configuration", "reconcile", "extraction", "embedding", "graph_validation", "write"}:
        raise ValueError(f"unknown ingestion failure stage: {stage}")
    chain = exception_chain(error)
    if stage in {"input", "configuration"}:
        return FailureDecision(stage, False)
    if any(isinstance(e, IngestionStateError) for e in chain):
        return FailureDecision("unsafe_graph_state", False)
    statuses = [getattr(e, "status_code", None) for e in chain]
    if any(isinstance(e, AuthError) for e in chain) or any(s in (401, 403) for s in statuses):
        return FailureDecision("authentication", False)
    if any(isinstance(e, ClientError) and not isinstance(e, TransientError) for e in chain):
        return FailureDecision("database_client", False)
    # Permanent HTTP errors override an incidental nested connection error.
    if any(isinstance(s, int) and 400 <= s < 500 and s not in (408, 429) for s in statuses):
        return FailureDecision("invalid_request", False)
    if 429 in statuses or any(isinstance(e, RateLimitError) for e in chain):
        return FailureDecision("rate_limit", True)
    if any(s == 408 or isinstance(s, int) and 500 <= s <= 599 for s in statuses):
        return FailureDecision("temporary_service", True)
    if any(isinstance(e, (TimeoutError, APITimeoutError, httpx.TimeoutException)) for e in chain):
        return FailureDecision("timeout", True)
    if any(isinstance(e, (ConnectionError, APIConnectionError, httpx.NetworkError,
                          ServiceUnavailable, SessionExpired, TransientError, RetryableError, httpx.RemoteProtocolError)) for e in chain):
        return FailureDecision("connectivity_or_transient_database", True)
    if stage in {"extraction", "graph_validation"} and any(
        isinstance(e, (ValidationError, json.JSONDecodeError, InvalidJSONError))
        or stage == "graph_validation" and isinstance(e, ValueError) for e in chain
    ):
        return FailureDecision("malformed_model_output", True)
    return FailureDecision("unknown", False)


class PipelineStageError(RuntimeError):
    """Typed component boundary preserving the original exception as its cause."""

    def __init__(self, stage: Stage, message: str):
        super().__init__(message)
        self.stage = stage


def failure_stage(error: BaseException) -> Stage:
    from neo4j.exceptions import Neo4jError
    from neo4j_graphrag.exceptions import EmbeddingsGenerationError

    chain = exception_chain(error)
    for cause in chain:
        if isinstance(cause, PipelineStageError):
            return cause.stage
    if any(isinstance(cause, EmbeddingsGenerationError) for cause in chain):
        return "embedding"
    if any(isinstance(cause, (Neo4jError, ServiceUnavailable, SessionExpired)) for cause in chain):
        return "write"
    return "extraction"


def attempt_configuration(neo4j_settings, model_settings) -> dict[str, str | int]:
    """Whitelist scalar operational settings; never serialize settings objects."""
    from graphrag_pipeline.extraction.idempotency import CONTRACT_VERSION

    result: dict[str, str | int] = {"ingestion_contract_version": CONTRACT_VERSION}
    provider = type(model_settings).__name__.removesuffix("Settings").lower()
    if provider in {"ollama", "openrouter"}:
        result["provider"] = provider
    for name in ("llm_model", "embedding_model", "embedding_dimensions", "llm_max_tokens"):
        value = getattr(model_settings, name, None)
        if type(value) in (str, int):
            result[name] = value
    database = getattr(neo4j_settings, "database", None)
    if type(database) is str:
        result["database"] = database
    return result
