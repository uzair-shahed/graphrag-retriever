"""Grounded answer-generation boundary for retrieved GraphRAG evidence.

Last modified: 2026-10-06

Purpose of this file:
    Send a validated question and Neo4j-returned evidence to the configured
    GraphRAG-supported LLM, then enforce answer and citation requirements in
    the application's ``QueryResponse`` contract.

Files that use this file:
    The query command and retrieval-test program will build an answer LLM with
    ``build_answer_llm`` and call ``generate_answer``. Unit tests use fake LLMs
    and never make paid or network calls.

Caveats:
    Retrieval is performed by ``graphrag_pipeline.retrieval``. This boundary
    intentionally does not pass MuSiQue gold labels, supporting-fact labels, or
    decomposition fields to the provider.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictStr, ValidationError
from neo4j_graphrag.llm import OllamaLLM, OpenAILLM

from graphrag_pipeline.config import OllamaSettings, OpenRouterSettings, is_reasoning_model
from graphrag_pipeline.diagnostics import save_invalid_llm_response
from graphrag_pipeline.query import QueryResponse, QueryRequest
from graphrag_pipeline.retrieval import RetrievalResult

logger = logging.getLogger(__name__)


class AnswerGenerationError(RuntimeError):
    """Raised when the answer provider returns an unusable response."""


class AnswerWire(BaseModel):
    """Minimal strict-schema wire model for OpenAI/OpenRouter structured outputs."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    citations: list[str]


class GeneratedAnswer(BaseModel):
    """Strict provider output required for a grounded answer."""

    model_config = ConfigDict(extra="forbid")

    answer: StrictStr = Field(min_length=1)
    citations: list[StrictStr] = Field(min_length=1)


SYSTEM_ANSWER_INSTRUCTION = (
    "Answer the question using only the graph evidence below. "
    "Return valid JSON with exactly two fields: answer (string) and "
    "citations (array of string identifiers). Every citation must identify "
    "evidence used in the answer matching the bracketed [id] labels. "
    "If the evidence does not support an answer, do not guess."
)


@dataclass(frozen=True)
class AnswerSettings:
    """Bounded answer-generation configuration."""

    retrieval_mode: str
    max_evidence_chunks: int = 5

    def __post_init__(self) -> None:
        if not self.retrieval_mode:
            raise ValueError("retrieval_mode must not be empty")
        if self.max_evidence_chunks < 1:
            raise ValueError("max_evidence_chunks must be positive")


def build_answer_llm(
    model_settings: OpenRouterSettings | OllamaSettings,
) -> Any:
    """Build the configured Neo4j GraphRAG-supported answer LLM.

    Args:
        model_settings: Existing OpenRouter or Ollama model configuration.

    Returns:
        A synchronous GraphRAG LLM wrapper with ``invoke``.
    """
    if isinstance(model_settings, OpenRouterSettings):
        provider_dict: dict[str, Any] = {
            "require_parameters": model_settings.require_parameters,
            "allow_fallbacks": model_settings.allow_fallbacks,
        }
        if model_settings.provider_order:
            provider_dict["order"] = model_settings.provider_order

        model_params: dict[str, Any] = {
            "temperature": model_settings.temperature,
            "max_tokens": model_settings.llm_max_tokens,
            "extra_body": {
                "provider": provider_dict,
            },
        }
        if model_settings.seed is not None:
            model_params["seed"] = model_settings.seed
        if is_reasoning_model(model_settings.llm_model):
            model_params["reasoning_effort"] = "none"

        logger.info(
            "Configured answer OpenAILLM: model=%s, profile=%s, temp=%s, seed=%s, provider_order=%s",
            model_settings.llm_model,
            model_settings.profile,
            model_settings.temperature,
            model_settings.seed,
            model_settings.provider_order,
        )
        return OpenAILLM(
            model_name=model_settings.llm_model,
            model_params=model_params,
            base_url=model_settings.base_url,
            api_key=model_settings.api_key,
        )
    return OllamaLLM(
        model_name=model_settings.llm_model,
        model_params={"options": {"temperature": 0}},
        host=model_settings.host,
    )


def _prompt(request: QueryRequest, retrieval: RetrievalResult, limit: int) -> str:
    """Build a user prompt containing the question, graph evidence, and bridging entity names."""
    evidence = retrieval.evidence[:limit]
    all_bridges = sorted({b for item in evidence for b in item.graph_references})
    bridge_prefix = f"Bridging entities: {', '.join(all_bridges)}\n\n" if all_bridges else ""
    context = "\n\n".join(
        f"[{item.paragraph_hash or item.chunk_id}]"
        + (f" (bridged via: {', '.join(item.graph_references)})" if item.graph_references else "")
        + f" {item.text}"
        for item in evidence
    )
    return (
        f"Question: {request.question}\n\n"
        f"{bridge_prefix}"
        f"Graph evidence:\n{context}"
    )


def _response_metadata(request: QueryRequest) -> dict[str, Any]:
    """Return request provenance fields shared by all answer outcomes."""
    return {
        "trace_id": request.trace_id,
        "question": request.question,
        "origin": request.origin,
        "record_id": request.record_id,
        "source_file": request.source_file,
        "source_line": request.source_line,
        "preload": request.preload,
    }


def _parse_provider_output(
    raw_response: Any,
    *,
    diagnostics_dir: Path = Path("debugging_logs"),
) -> GeneratedAnswer:
    """Parse provider content and normalize malformed JSON as an application error."""
    content = getattr(raw_response, "content", raw_response)
    if not isinstance(content, str) or not content.strip():
        raise AnswerGenerationError("answer provider returned empty content")
    try:
        return GeneratedAnswer.model_validate_json(content)
    except (ValidationError, ValueError) as error:
        save_invalid_llm_response(error, content, directory=diagnostics_dir)
        raise AnswerGenerationError("answer provider returned malformed JSON output") from error


def generate_answer(
    request: QueryRequest,
    retrieval: RetrievalResult,
    llm: Any,
    *,
    answer_settings: AnswerSettings | None = None,
    diagnostics_dir: Path = Path("debugging_logs"),
) -> QueryResponse:
    """Generate a grounded answer or explicit insufficient-evidence result.

    Args:
        request: Validated question and source provenance.
        retrieval: Graph-backed evidence returned by the retrieval adapter.
        llm: Configured GraphRAG LLM wrapper with synchronous ``invoke``.
        answer_settings: Retrieval mode and evidence bound for the response.
        diagnostics_dir: Directory where invalid LLM responses are saved.

    Returns:
        A provenance-preserving ``QueryResponse`` with grounded citations or a
        non-success status for empty, malformed, or failed generation.
    """
    settings = answer_settings or AnswerSettings(retrieval_mode=retrieval.mode)
    metadata = _response_metadata(request)
    evidence = retrieval.evidence[: settings.max_evidence_chunks]
    if not evidence:
        return QueryResponse(
            **metadata,
            status="insufficient_evidence",
            retrieval_mode=settings.retrieval_mode,
            predicted_support_idxs=[],
            predicted_answerable=False,
        )

    try:
        messages = [
            {"role": "system", "content": SYSTEM_ANSWER_INSTRUCTION},
            {"role": "user", "content": _prompt(request, retrieval, settings.max_evidence_chunks)},
        ]
        if isinstance(llm, OllamaLLM):
            raw_response = llm.invoke(
                input=messages,
                format=AnswerWire.model_json_schema(),
            )
        else:
            raw_response = llm.invoke(
                input=messages,
                response_format=AnswerWire,
            )
        generated = _parse_provider_output(raw_response, diagnostics_dir=diagnostics_dir)
        if not generated.answer.strip():
            raise AnswerGenerationError("answer provider returned an empty answer")

        valid_citations: dict[str, EvidenceChunk] = {}
        for item in evidence:
            if item.chunk_id:
                valid_citations[item.chunk_id] = item
            if item.paragraph_hash:
                valid_citations[item.paragraph_hash] = item
            if item.paragraph_idx is not None:
                valid_citations[str(item.paragraph_idx)] = item

        invalid_citations = [
            citation for citation in generated.citations
            if citation not in valid_citations
        ]
        if invalid_citations:
            logger.warning(
                "Rejected non-retrieved citations %r for question %r (valid tokens: %r)",
                invalid_citations,
                request.question,
                list(valid_citations.keys()),
            )
            raise AnswerGenerationError(
                f"answer cited chunks outside retrieved evidence: {invalid_citations}"
            )

        cited_idxs: list[int] = []
        normalized_citations: list[str] = []
        for citation in generated.citations:
            matched_chunk = valid_citations[citation]
            norm_id = matched_chunk.paragraph_hash or matched_chunk.chunk_id
            if norm_id not in normalized_citations:
                normalized_citations.append(norm_id)
            if matched_chunk.paragraph_idx is not None and matched_chunk.paragraph_idx not in cited_idxs:
                cited_idxs.append(matched_chunk.paragraph_idx)

        cited_idxs.sort()

        return QueryResponse(
            **metadata,
            status="succeeded",
            answer=generated.answer,
            retrieval_mode=settings.retrieval_mode,
            evidence=evidence,
            citations=normalized_citations,
            predicted_support_idxs=cited_idxs,
            predicted_answerable=True,
        )
    except AnswerGenerationError as error:
        return QueryResponse(
            **metadata,
            status="failed",
            retrieval_mode=settings.retrieval_mode,
            evidence=evidence,
            error=str(error),
            predicted_support_idxs=[],
            predicted_answerable=False,
        )
    except Exception as error:  # noqa: BLE001 - provider errors become failed responses
        return QueryResponse(
            **metadata,
            status="failed",
            retrieval_mode=settings.retrieval_mode,
            evidence=evidence,
            error=f"answer generation failed: {error}",
            predicted_support_idxs=[],
            predicted_answerable=False,
        )
