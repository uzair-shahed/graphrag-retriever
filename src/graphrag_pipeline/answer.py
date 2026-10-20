"""Grounded answer-generation boundary for retrieved GraphRAG evidence.

Last modified: 2026-10-17

Purpose of this file:
    Send a validated question and Neo4j-returned evidence to the configured
    GraphRAG-supported LLM, then enforce answer and citation requirements in
    the application's ``QueryResponse`` contract. The same JSON contract and
    ``validate_generated_answer`` check apply whether the answer came from this
    module's own LLM call (VectorRAG) or from ``GraphRAG.search`` (GraphRAG).

Files that use this file:
    The query command and retrieval-test program will build an answer LLM with
    ``build_answer_llm`` and call ``generate_answer``. Unit tests use fake LLMs
    and never make paid or network calls.

Caveats:
    Retrieval is performed by ``graphrag_pipeline.retrieval``. This boundary
    intentionally does not pass MuSiQue gold labels, supporting-fact labels, or
    decomposition fields to the provider. Provider output is parsed strictly
    except that one surrounding markdown code fence is accepted. The model
    can explicitly abstain with ``insufficient_evidence`` while preserving
    retrieved evidence. Older successful two-field output remains accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, ValidationError, model_validator
from neo4j_graphrag.llm import OpenAILLM

from graphrag_pipeline.answer_context import ANSWER_PROMPT_TEMPLATE, render_evidence_context
from graphrag_pipeline.config import OllamaSettings, OpenRouterSettings
from graphrag_pipeline.ollama_json_schema import OllamaJSONSchemaLLM
from graphrag_pipeline.query import EvidenceChunk, QueryResponse, QueryRequest
from graphrag_pipeline.retrieval import RetrievalResult


class AnswerGenerationError(RuntimeError):
    """Raised when the answer provider returns an unusable response."""


class GeneratedAnswer(BaseModel):
    """Strict provider output required for a grounded answer."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["answered", "insufficient_evidence"] = "answered"
    answer: StrictStr | None = Field(min_length=1)
    citations: list[StrictStr]

    @model_validator(mode="after")
    def validate_outcome(self) -> "GeneratedAnswer":
        if self.status == "answered" and (self.answer is None or not self.citations):
            raise ValueError("answered output requires an answer and citations")
        if self.status == "insufficient_evidence" and (self.answer is not None or self.citations):
            raise ValueError("abstention requires null answer and empty citations")
        return self


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
        return OpenAILLM(
            model_name=model_settings.llm_model,
            model_params={
                "temperature": 0,
                "reasoning_effort": "none",
                "max_tokens": model_settings.llm_max_tokens,
                "extra_body": {
                    "provider": {
                        "require_parameters": model_settings.require_parameters,
                        "allow_fallbacks": model_settings.allow_fallbacks,
                    }
                },
            },
            base_url=model_settings.base_url,
            api_key=model_settings.api_key,
        )
    return OllamaJSONSchemaLLM(
        model_name=model_settings.llm_model,
        response_model=GeneratedAnswer,
        model_params={"options": {"temperature": 0, "num_predict": model_settings.llm_max_tokens}},
        host=model_settings.host,
    )


def _prompt(request: QueryRequest, evidence: list[EvidenceChunk]) -> str:
    """Build the VectorRAG prompt from the shared template and bounded evidence."""
    return ANSWER_PROMPT_TEMPLATE.format(
        query_text=request.question,
        context=render_evidence_context(evidence),
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


_CODE_FENCE = re.compile(r"^```(?:json)?\s*(?P<body>.*?)\s*```$", re.DOTALL | re.IGNORECASE)


def _parse_provider_output(raw_response: Any) -> GeneratedAnswer:
    """Parse provider content and normalize malformed JSON as an application error."""
    content = getattr(raw_response, "content", raw_response)
    if not isinstance(content, str) or not content.strip():
        raise AnswerGenerationError("answer provider returned empty content")
    fenced = _CODE_FENCE.match(content.strip())
    if fenced:
        content = fenced.group("body")
    try:
        return GeneratedAnswer.model_validate_json(content)
    except (ValidationError, ValueError) as error:
        raise AnswerGenerationError("answer provider returned malformed JSON output") from error


def validate_generated_answer(raw_response: Any, evidence: list[EvidenceChunk]) -> GeneratedAnswer:
    """Apply the answer contract shared by VectorRAG and GraphRAG.

    Args:
        raw_response: Provider response object or raw JSON text.
        evidence: The exact bounded evidence the model was shown.

    Returns:
        The parsed answer with duplicate citations removed, order preserved.

    Raises:
        AnswerGenerationError: For empty or malformed output, a blank answer,
            or any citation that is not a chunk ID in ``evidence``.
    """
    generated = _parse_provider_output(raw_response)
    if generated.status == "answered" and not generated.answer.strip():
        raise AnswerGenerationError("answer provider returned an empty answer")
    evidence_ids = {item.chunk_id for item in evidence}
    invalid_citations = [citation for citation in generated.citations if citation not in evidence_ids]
    if invalid_citations:
        raise AnswerGenerationError(
            f"answer cited chunks outside retrieved evidence: {invalid_citations}"
        )
    return generated.model_copy(update={"citations": list(dict.fromkeys(generated.citations))})


def generate_answer(
    request: QueryRequest,
    retrieval: RetrievalResult,
    llm: Any,
    *,
    answer_settings: AnswerSettings | None = None,
) -> QueryResponse:
    """Generate a grounded answer or explicit insufficient-evidence result.

    Args:
        request: Validated question and source provenance.
        retrieval: Graph-backed evidence returned by the retrieval adapter.
        llm: Configured GraphRAG LLM wrapper with synchronous ``invoke``.
        answer_settings: Retrieval mode and evidence bound for the response.

    Returns:
        A provenance-preserving ``QueryResponse`` with grounded citations or a
        non-success status for empty, malformed, or failed generation.
    """
    settings = answer_settings or AnswerSettings(retrieval_mode=retrieval.mode)
    metadata = _response_metadata(request)
    metadata["retrieval_backend"] = retrieval.backend
    evidence = retrieval.evidence[: settings.max_evidence_chunks]
    if not evidence:
        return QueryResponse(
            **metadata,
            status="insufficient_evidence",
            retrieval_mode=settings.retrieval_mode,
        )

    try:
        if retrieval.generated_answer is not None:
            if len(retrieval.evidence) > settings.max_evidence_chunks:
                # The pre-generated answer already saw every retrieved chunk.
                raise AnswerGenerationError(
                    f"generated answer used {len(retrieval.evidence)} chunks, exceeding "
                    f"max_evidence_chunks={settings.max_evidence_chunks}"
                )
            raw_response = retrieval.generated_answer
        else:
            raw_response = llm.invoke(
                input=_prompt(request, evidence),
                response_format={"type": "json_object"},
            )
        generated = validate_generated_answer(raw_response, evidence)
        return QueryResponse(
            **metadata,
            status="succeeded" if generated.status == "answered" else "insufficient_evidence",
            answer=generated.answer,
            retrieval_mode=settings.retrieval_mode,
            evidence=evidence,
            citations=generated.citations,
        )
    except AnswerGenerationError as error:
        return QueryResponse(
            **metadata,
            status="failed",
            retrieval_mode=settings.retrieval_mode,
            evidence=evidence,
            error=str(error),
        )
    except Exception as error:  # noqa: BLE001 - provider errors become failed responses
        return QueryResponse(
            **metadata,
            status="failed",
            retrieval_mode=settings.retrieval_mode,
            evidence=evidence,
            error=f"answer generation failed: {error}",
        )
