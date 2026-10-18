"""Deterministic tests for the grounded answer-generation boundary."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from graphrag_pipeline.answer import (
    AnswerSettings,
    GeneratedAnswer,
    build_answer_llm,
    generate_answer,
)
from graphrag_pipeline.config import OllamaSettings, OpenRouterSettings
from graphrag_pipeline.query import EvidenceChunk, QueryRequest
from graphrag_pipeline.retrieval import RetrievalResult


def request() -> QueryRequest:
    """Return a request with source provenance for response assertions."""
    return QueryRequest(
        question="Which company acquired Northwind Labs?",
        trace_id="trace-answer-1",
        origin="musique",
        record_id="record-1",
        source_file="fixture.jsonl",
        source_line=2,
        preload=False,
    )


def retrieval(*chunks: EvidenceChunk) -> RetrievalResult:
    """Build a deterministic retrieval result."""
    return RetrievalResult(
        trace_id="trace-answer-1",
        question=request().question,
        mode="vector_cypher",
        evidence=list(chunks),
    )


def chunk(chunk_id: str = "chunk-1") -> EvidenceChunk:
    """Build one graph evidence chunk."""
    return EvidenceChunk(
        chunk_id=chunk_id,
        text="Aster Systems acquired Northwind Labs.",
        document_id="document-1",
        source_file="fixture.jsonl",
        source_line=2,
    )


def test_generate_answer_returns_grounded_answer_and_citations() -> None:
    """Valid provider JSON produces a cited successful response."""
    llm = MagicMock()
    llm.invoke.return_value = SimpleNamespace(
        content='{"answer":"Aster Systems.","citations":["chunk-1"]}'
    )

    response = generate_answer(request(), retrieval(chunk()), llm)

    assert response.status == "succeeded"
    assert response.answer == "Aster Systems."
    assert response.citations == ["chunk-1"]
    assert response.evidence[0].chunk_id == "chunk-1"
    assert response.trace_id == "trace-answer-1"
    prompt = llm.invoke.call_args.kwargs["input"]
    assert request().question in prompt
    assert "chunk-1" in prompt
    assert "is_supporting" not in prompt


def test_generate_answer_marks_empty_evidence_insufficient_without_provider_call() -> None:
    """Empty retrieval cannot become an invented answer or provider call."""
    llm = MagicMock()

    response = generate_answer(request(), retrieval(), llm)

    assert response.status == "insufficient_evidence"
    assert response.answer is None
    llm.invoke.assert_not_called()


def graphrag_retrieval(generated_answer: str | None, *chunks: EvidenceChunk) -> RetrievalResult:
    """Build a GraphRAG retrieval result carrying the managed answer text."""
    return RetrievalResult(
        trace_id="trace-1",
        question="Which company acquired Northwind Labs?",
        mode="graphrag",
        backend="graphrag",
        generated_answer=generated_answer,
        evidence=list(chunks),
    )


def test_generate_answer_uses_graph_rag_answer_without_second_llm_call() -> None:
    """GraphRAG's JSON answer is validated and cites only what it declares."""
    llm = MagicMock()
    retrieval_result = graphrag_retrieval(
        '{"answer":"Aster Systems","citations":["chunk-2"]}', chunk("chunk-1"), chunk("chunk-2")
    )

    response = generate_answer(request(), retrieval_result, llm)

    assert response.status == "succeeded"
    assert response.answer == "Aster Systems"
    assert response.citations == ["chunk-2"]
    assert [item.chunk_id for item in response.evidence] == ["chunk-1", "chunk-2"]
    assert response.retrieval_backend == "graphrag"
    llm.invoke.assert_not_called()


def test_generate_answer_rejects_provider_citation_outside_evidence() -> None:
    """Citations must refer to returned graph chunks."""
    llm = MagicMock()
    llm.invoke.return_value = SimpleNamespace(
        content='{"answer":"Aster Systems.","citations":["unknown"]}'
    )

    response = generate_answer(request(), retrieval(chunk()), llm)

    assert response.status == "failed"
    assert "outside retrieved evidence" in response.error


def test_generate_answer_rejects_malformed_provider_output() -> None:
    """Malformed provider output is an explicit failure, never a success."""
    llm = MagicMock()
    llm.invoke.return_value = SimpleNamespace(content="not json")

    response = generate_answer(request(), retrieval(chunk()), llm)

    assert response.status == "failed"
    assert response.answer is None
    assert "malformed JSON" in response.error


def test_generate_answer_rejects_whitespace_only_answer() -> None:
    """Whitespace-only output is not a grounded answer."""
    llm = MagicMock()
    llm.invoke.return_value = SimpleNamespace(
        content='{"answer":"   ","citations":["chunk-1"]}'
    )

    response = generate_answer(request(), retrieval(chunk()), llm)

    assert response.status == "failed"
    assert "empty answer" in response.error


def test_generate_answer_exposes_provider_failure() -> None:
    """Provider exceptions preserve evidence and become failed responses."""
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("provider unavailable")

    response = generate_answer(request(), retrieval(chunk()), llm)

    assert response.status == "failed"
    assert "provider unavailable" in response.error
    assert response.evidence[0].chunk_id == "chunk-1"


def test_generate_answer_respects_evidence_bound() -> None:
    """Only the configured evidence prefix is sent and cited."""
    llm = MagicMock()
    llm.invoke.return_value = SimpleNamespace(
        content='{"answer":"Aster Systems.","citations":["chunk-1"]}'
    )

    response = generate_answer(
        request(),
        retrieval(chunk("chunk-1"), chunk("chunk-2")),
        llm,
        answer_settings=AnswerSettings(retrieval_mode="vector", max_evidence_chunks=1),
    )

    assert [item.chunk_id for item in response.evidence] == ["chunk-1"]
    assert "chunk-2" not in llm.invoke.call_args.kwargs["input"]


def test_build_answer_llm_uses_existing_provider_settings() -> None:
    """Answer LLM construction follows the OpenRouter and Ollama conventions."""
    openrouter = OpenRouterSettings(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        llm_model="test-llm",
        embedding_model="test-embedding",
        embedding_dimensions=3,
        llm_max_tokens=100,
        require_parameters=True,
        allow_fallbacks=True,
    )
    assert build_answer_llm(openrouter).model_name == "test-llm"

    ollama = OllamaSettings(
        host="http://localhost:11434",
        llm_model="local-llm",
        embedding_model="local-embedding",
    )
    ollama_llm = build_answer_llm(ollama)
    assert ollama_llm.model_name == "local-llm"
    assert ollama_llm.response_schema == GeneratedAnswer.model_json_schema()


def test_generate_answer_rejects_pregenerated_answer_that_saw_more_than_the_bound() -> None:
    """A GraphRAG answer built from more chunks than the bound is a failure, not a truncated success."""
    llm = MagicMock()
    retrieval_result = graphrag_retrieval(
        '{"answer":"Answer that depended on chunk-2","citations":["chunk-2"]}',
        chunk("chunk-1"),
        chunk("chunk-2"),
    )

    response = generate_answer(
        request(),
        retrieval_result,
        llm,
        answer_settings=AnswerSettings(retrieval_mode="graphrag", max_evidence_chunks=1),
    )

    assert response.status == "failed"
    assert response.answer is None
    assert "exceeding max_evidence_chunks=1" in response.error


def test_generate_answer_keeps_pregenerated_answer_evidence_identical_to_retrieval() -> None:
    """Within the bound, response evidence is exactly the retrieved evidence."""
    retrieval_result = graphrag_retrieval(
        '{"answer":"Aster Systems","citations":["chunk-1","chunk-2"]}',
        chunk("chunk-1"),
        chunk("chunk-2"),
    )

    response = generate_answer(
        request(),
        retrieval_result,
        MagicMock(),
        answer_settings=AnswerSettings(retrieval_mode="graphrag", max_evidence_chunks=2),
    )

    assert [item.chunk_id for item in response.evidence] == ["chunk-1", "chunk-2"]
    assert response.citations == ["chunk-1", "chunk-2"]


GRAPH_BACKEND_OUTPUTS = {
    "empty": ("", "empty content"),
    "blank": ("   ", "empty content"),
    "malformed": ("The answer is Aster Systems.", "malformed JSON"),
    "missing_citations": ('{"answer":"Aster Systems","citations":[]}', "malformed JSON"),
    "extra_field": ('{"answer":"Aster","citations":["chunk-1"],"confidence":1}', "malformed JSON"),
    "blank_answer": ('{"answer":"  ","citations":["chunk-1"]}', "empty answer"),
    "unsupported_citation": ('{"answer":"Aster","citations":["chunk-9"]}', "outside retrieved evidence"),
}


@pytest.mark.parametrize("backend", ["vectorrag", "graphrag"])
@pytest.mark.parametrize("case", sorted(GRAPH_BACKEND_OUTPUTS))
def test_both_backends_reject_invalid_provider_output_as_failed(backend: str, case: str) -> None:
    """Invalid output becomes the same structured failure for both backends."""
    output, expected_error = GRAPH_BACKEND_OUTPUTS[case]
    llm = MagicMock()
    if backend == "graphrag":
        retrieval_result = graphrag_retrieval(output, chunk("chunk-1"))
    else:
        retrieval_result = retrieval(chunk("chunk-1"))
        llm.invoke.return_value = SimpleNamespace(content=output)

    response = generate_answer(request(), retrieval_result, llm)

    assert response.status == "failed"
    assert response.answer is None
    assert response.citations == []
    assert expected_error in response.error
    assert response.retrieval_backend == retrieval_result.backend
    assert [item.chunk_id for item in response.evidence] == ["chunk-1"]


@pytest.mark.parametrize("backend", ["vectorrag", "graphrag"])
def test_both_backends_mark_empty_evidence_insufficient_without_validation(backend: str) -> None:
    """Empty evidence is insufficient_evidence even if a provider produced an answer."""
    llm = MagicMock()
    retrieval_result = (
        graphrag_retrieval('{"answer":"invented","citations":["chunk-1"]}')
        if backend == "graphrag"
        else retrieval()
    )

    response = generate_answer(request(), retrieval_result, llm)

    assert response.status == "insufficient_evidence"
    assert response.answer is None
    assert response.citations == []
    llm.invoke.assert_not_called()


@pytest.mark.parametrize("backend", ["vectorrag", "graphrag"])
def test_both_backends_cite_only_the_declared_subset_without_duplicates(backend: str) -> None:
    """A subset citation is preserved and repeats collapse; unused chunks are not cited."""
    output = '{"answer":"Aster Systems","citations":["chunk-2","chunk-2"]}'
    llm = MagicMock()
    if backend == "graphrag":
        retrieval_result = graphrag_retrieval(output, chunk("chunk-1"), chunk("chunk-2"))
    else:
        retrieval_result = retrieval(chunk("chunk-1"), chunk("chunk-2"))
        llm.invoke.return_value = SimpleNamespace(content=output)

    response = generate_answer(request(), retrieval_result, llm)

    assert response.status == "succeeded"
    assert response.citations == ["chunk-2"]
    assert [item.chunk_id for item in response.evidence] == ["chunk-1", "chunk-2"]


def test_provider_output_may_be_wrapped_in_one_json_code_fence() -> None:
    """A single markdown fence around valid JSON is accepted for both backends."""
    fenced = '```json\n{"answer":"Aster Systems","citations":["chunk-1"]}\n```'

    response = generate_answer(request(), graphrag_retrieval(fenced, chunk()), MagicMock())

    assert response.status == "succeeded"
    assert response.citations == ["chunk-1"]


def test_vectorrag_prompt_uses_the_shared_template_and_bracketed_chunk_ids() -> None:
    """The VectorRAG prompt is the shared template filled with rendered evidence."""
    from graphrag_pipeline.answer import _prompt
    from graphrag_pipeline.answer_context import ANSWER_PROMPT_TEMPLATE, render_evidence_context

    evidence = [chunk("chunk-1"), chunk("chunk-2")]

    prompt = _prompt(request(), evidence)

    assert prompt == ANSWER_PROMPT_TEMPLATE.format(
        query_text=request().question, context=render_evidence_context(evidence)
    )
    assert "[chunk-1] Aster Systems acquired Northwind Labs." in prompt
    assert "[chunk-2]" in prompt
