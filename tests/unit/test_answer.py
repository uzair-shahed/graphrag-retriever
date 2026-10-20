"""Deterministic tests for the grounded answer-generation boundary."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from graphrag_pipeline.answer import AnswerSettings, AnswerWire, build_answer_llm, generate_answer
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
    messages = llm.invoke.call_args.kwargs["input"]
    assert isinstance(messages, list)
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert request().question in messages[1]["content"]
    assert "chunk-1" in messages[1]["content"]
    assert "is_supporting" not in messages[1]["content"]
    assert llm.invoke.call_args.kwargs["response_format"] is AnswerWire


def test_generate_answer_marks_empty_evidence_insufficient_without_provider_call() -> None:
    """Empty retrieval cannot become an invented answer or provider call."""
    llm = MagicMock()

    response = generate_answer(request(), retrieval(), llm)

    assert response.status == "insufficient_evidence"
    assert response.answer is None
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


def test_generate_answer_rejects_malformed_provider_output(tmp_path) -> None:
    """Malformed provider output is an explicit failure and saves raw reply."""
    llm = MagicMock()
    llm.invoke.return_value = SimpleNamespace(content="not json")

    response = generate_answer(request(), retrieval(chunk()), llm, diagnostics_dir=tmp_path)

    assert response.status == "failed"
    assert response.answer is None
    assert "malformed JSON" in response.error
    saved = list(tmp_path.glob("invalid_llm_response_*.txt"))
    assert len(saved) == 1
    assert saved[0].read_text(encoding="utf-8") == "not json"


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
    messages = llm.invoke.call_args.kwargs["input"]
    user_content = messages[1]["content"] if isinstance(messages, list) else str(messages)
    assert "chunk-2" not in user_content


def test_answer_wire_schema_is_strict_compatible() -> None:
    """AnswerWire schema has required fields, no extra properties, and strict object form."""
    schema = AnswerWire.model_json_schema()
    assert schema.get("type") == "object"
    assert schema.get("additionalProperties") is False
    assert set(schema.get("required", [])) == {"answer", "citations"}
    assert schema["properties"]["answer"]["type"] == "string"
    assert schema["properties"]["citations"]["type"] == "array"
    assert schema["properties"]["citations"]["items"]["type"] == "string"


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
    assert build_answer_llm(ollama).model_name == "local-llm"


def test_openai_llm_translates_answer_wire_to_strict_json_schema() -> None:
    """OpenAILLM translates AnswerWire into OpenAI strict json_schema payload."""
    openrouter = OpenRouterSettings(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        llm_model="openai/gpt-4o-mini",
        embedding_model="openai/text-embedding-3-small",
        embedding_dimensions=1536,
        llm_max_tokens=100,
        require_parameters=True,
        allow_fallbacks=False,
    )
    llm = build_answer_llm(openrouter)
    llm.client = MagicMock()
    llm.client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"answer":"Northwind Labs","citations":["chunk-1"]}'
                )
            )
        ],
        usage=None,
    )

    response = generate_answer(request(), retrieval(chunk()), llm)

    assert response.status == "succeeded"
    assert response.answer == "Northwind Labs"
    create_kwargs = llm.client.chat.completions.create.call_args.kwargs
    response_format = create_kwargs["response_format"]
    assert response_format["type"] == "json_schema"
    json_schema = response_format["json_schema"]
    assert json_schema["strict"] is True
    assert json_schema["name"] == "AnswerWire"
    schema = json_schema["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"answer", "citations"}
    assert create_kwargs["temperature"] == 0.0
    assert create_kwargs["seed"] == 20261018
    assert create_kwargs["extra_body"]["provider"]["order"] == ["openai"]
    assert create_kwargs["extra_body"]["provider"]["allow_fallbacks"] is False
    assert create_kwargs["extra_body"]["provider"]["require_parameters"] is True
    assert "reasoning_effort" not in create_kwargs


def test_ollama_llm_passes_format_option_without_openai_response_format() -> None:
    """OllamaLLM receives format option with JSON schema and no OpenAI response_format."""
    ollama_settings = OllamaSettings(
        host="http://localhost:11434",
        llm_model="local-llm",
        embedding_model="local-embedding",
    )
    llm = build_answer_llm(ollama_settings)
    mock_client = MagicMock()
    mock_client.chat.return_value = SimpleNamespace(
        message=SimpleNamespace(
            content='{"answer":"Local Aster Systems.","citations":["chunk-1"]}'
        ),
        prompt_eval_count=12,
        eval_count=6,
    )
    llm.client = mock_client

    response = generate_answer(request(), retrieval(chunk()), llm)

    assert response.status == "succeeded"
    assert response.answer == "Local Aster Systems."
    assert response.citations == ["chunk-1"]

    chat_kwargs = mock_client.chat.call_args.kwargs
    assert "format" in chat_kwargs
    assert chat_kwargs["format"] == AnswerWire.model_json_schema()
    assert "response_format" not in chat_kwargs


def test_generate_answer_maps_paragraph_hashes_to_predicted_support_idxs() -> None:
    """Paragraph hashes cited by the model map to predicted_support_idxs and predicted_answerable=True."""
    llm = MagicMock()
    llm.invoke.return_value = SimpleNamespace(
        content='{"answer":"Paris.","citations":["hash_chunk_2", "hash_chunk_0"]}'
    )

    c0 = EvidenceChunk(
        chunk_id="musique:paragraph:hash_chunk_0",
        paragraph_hash="hash_chunk_0",
        paragraph_idx=10,
        text="Chunk 0 text",
    )
    c1 = EvidenceChunk(
        chunk_id="musique:paragraph:hash_chunk_1",
        paragraph_hash="hash_chunk_1",
        paragraph_idx=5,
        text="Chunk 1 text",
    )
    c2 = EvidenceChunk(
        chunk_id="musique:paragraph:hash_chunk_2",
        paragraph_hash="hash_chunk_2",
        paragraph_idx=18,
        text="Chunk 2 text",
        graph_references=["Gong Band"],
    )

    response = generate_answer(request(), retrieval(c0, c1, c2), llm)

    assert response.status == "succeeded"
    assert response.answer == "Paris."
    assert response.predicted_answerable is True
    # Mapped and sorted paragraph indices
    assert response.predicted_support_idxs == [10, 18]
    assert response.citations == ["hash_chunk_2", "hash_chunk_0"]

    pred = response.to_prediction()
    assert pred["id"] == "record-1"
    assert pred["predicted_answer"] == "Paris."
    assert pred["predicted_support_idxs"] == [10, 18]
    assert pred["predicted_answerable"] is True


def test_generate_answer_prompt_includes_bridging_entities() -> None:
    """Prompt sent to LLM carries top paragraphs and bridging entity names."""
    llm = MagicMock()
    llm.invoke.return_value = SimpleNamespace(
        content='{"answer":"Paris.","citations":["hash_c0"]}'
    )

    c0 = EvidenceChunk(
        chunk_id="musique:paragraph:hash_c0",
        paragraph_hash="hash_c0",
        paragraph_idx=0,
        text="Steve Hillage was in Gong.",
        graph_references=["Gong", "Steve Hillage"],
    )
    c1 = EvidenceChunk(
        chunk_id="musique:paragraph:hash_c1",
        paragraph_hash="hash_c1",
        paragraph_idx=1,
        text="Gong formed in Paris.",
        graph_references=["Gong"],
    )

    response = generate_answer(request(), retrieval(c0, c1), llm)
    assert response.status == "succeeded"

    messages = llm.invoke.call_args.kwargs["input"]
    user_prompt = messages[1]["content"]

    assert "Bridging entities: Gong, Steve Hillage" in user_prompt
    assert "[hash_c0] [bridged via: Gong, Steve Hillage] Steve Hillage was in Gong." in user_prompt
    assert "[hash_c1] [bridged via: Gong] Gong formed in Paris." in user_prompt


def test_generate_answer_unretrieved_citation_rejected_and_recorded() -> None:
    """A citation for an unretrieved id is rejected, recorded in error, and fails loudly."""
    llm = MagicMock()
    llm.invoke.return_value = SimpleNamespace(
        content='{"answer":"Invented answer.","citations":["hallucinated_hash_999"]}'
    )

    c0 = EvidenceChunk(
        chunk_id="musique:paragraph:hash_c0",
        paragraph_hash="hash_c0",
        paragraph_idx=0,
        text="Real evidence.",
    )

    response = generate_answer(request(), retrieval(c0), llm)

    assert response.status == "failed"
    assert "answer cited chunks outside retrieved evidence" in response.error
    assert "hallucinated_hash_999" in response.error
    assert response.predicted_answerable is False
    assert response.predicted_support_idxs == []

    pred = response.to_prediction()
    assert pred["predicted_answerable"] is False
    assert pred["predicted_support_idxs"] == []


