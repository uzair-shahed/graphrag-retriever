"""Behavioral tests for selection, existing-graph evaluation and paired scoring."""

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic import ValidationError

from graphrag_pipeline.answer import GeneratedAnswer, generate_answer
from graphrag_pipeline.evaluation.command import build_parser, execute, selection_from_args
from graphrag_pipeline.evaluation.providers import EvaluatorSettings, build_metrics
from graphrag_pipeline.evaluation.records import DEFAULT_DATASET, GoldRecord, load_cases
from graphrag_pipeline.evaluation.runner import run_evaluation
from graphrag_pipeline.evaluation.runtime import EvaluationSettings, graph_readiness, select_context
from graphrag_pipeline.evaluation.scoring import evidence_metrics, gold_metrics, triad_scores
from graphrag_pipeline.query import EvidenceChunk, GraphNode, QueryResponse
from graphrag_pipeline.retrieval import RetrievalResult


@pytest.fixture
def source(tmp_path):
    records = [json.loads(s) for s in Path("tests/fixtures/musique/valid_records.jsonl").read_text().splitlines()]
    path = tmp_path / DEFAULT_DATASET.name
    path.write_text("\n" + "\n\n".join(json.dumps(r) for r in records) + "\n")
    return path


@pytest.fixture
def case(source):
    return load_cases(source, records=1)[0]


def evidence(case):
    return [EvidenceChunk(chunk_id=f"musique:{case.gold.id}:paragraph:{p.idx}", text=p.paragraph_text) for p in case.gold.paragraphs]


def response(case, status="succeeded", answer="Boston", citations=None):
    return QueryResponse(status=status, trace_id="t", question=case.gold.question, retrieval_mode="vector",
        answer=answer if status == "succeeded" else None, citations=citations or ([evidence(case)[1].chunk_id] if status == "succeeded" else []))


def test_range_matches_ingestion_and_preserves_actual_source_lines(source):
    cases = load_cases(source, start_record=2, records=2)
    assert [c.gold.id for c in cases] == ["fixture_valid_alias_002", "fixture_valid_unanswerable_003"]
    assert [c.source_line for c in cases] == [4, 6]
    assert len(load_cases(source, start_record=3, records=20)) == 1
    assert load_cases(source, record_ids=[cases[1].gold.id, cases[0].gold.id])[0].gold.id == cases[1].gold.id


@pytest.mark.parametrize("kwargs", [{"start_record": 0}, {"records": 0}, {"start_record": 4},
    {"record_ids": ["unknown"]}, {"record_ids": ["a", "a"]}, {"record_ids": ["a"], "records": 1}])
def test_invalid_selection_fails_before_network(source, kwargs):
    with pytest.raises(ValueError):
        load_cases(source, **kwargs)


def test_only_requested_dev_dataset_is_supported(tmp_path):
    with pytest.raises(ValueError, match="only"):
        load_cases(tmp_path / "musique_ans_v1.0_test.jsonl")


def test_duplicate_dataset_ids_rejected(source):
    source.write_text(source.read_text() + source.read_text())
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(source)


def test_gold_stays_out_of_query_and_extraction(case):
    payload = case.query("t").model_dump()
    extraction = [r.model_dump() for r in case.extraction_requests()]
    for label in ("answer", "answer_aliases", "answerable", "question_decomposition", "is_supporting"):
        assert label not in payload
        assert f'"{label}"' not in json.dumps(extraction)
    assert payload["question"] == case.gold.question
    assert case.extraction_requests()[0].chunk_id == evidence(case)[0].chunk_id


def test_invalid_hop_support_rejected(case):
    raw = case.gold.model_dump()
    raw["question_decomposition"][0]["paragraph_support_idx"] = 999
    with pytest.raises(ValidationError, match="missing paragraph"):
        GoldRecord.model_validate(raw)


def test_parser_compatible_with_loader_and_query_aliases(source):
    parser = build_parser()
    for command in ("run", "prepare", "check"):
        for flag in ("--records", "--max-records"):
            args = parser.parse_args([command, "--dataset-path", str(source), "--start-record", "2", flag, "1", "--dry-run"])
            assert selection_from_args(args)[0].gold.id == "fixture_valid_alias_002"
            code, result = asyncio.run(execute(args))
            assert code == 0 and result["records_selected"] == 1
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--records", "2", "--max-records", "3"])
    with pytest.raises(ValueError, match="cannot be combined"):
        selection_from_args(parser.parse_args(["run", "--record-id", "a", "--start-record", "1"]))


@pytest.mark.parametrize("backend", ["vectorrag", "graphrag"])
def test_explicit_abstention_with_evidence_for_both_backends(case, backend):
    llm = MagicMock()
    raw = '{"status":"insufficient_evidence","answer":null,"citations":[]}'
    llm.invoke.return_value = SimpleNamespace(content=raw)
    retrieved = RetrievalResult("t", case.gold.question, "vector", evidence(case), backend, raw if backend == "graphrag" else None)
    result = generate_answer(case.query("t", backend), retrieved, llm)
    assert result.status == "insufficient_evidence" and result.answer is None and result.citations == []
    assert result.evidence == evidence(case)
    if backend == "graphrag":
        llm.invoke.assert_not_called()


@pytest.mark.parametrize("payload", [
    {"status": "answered", "answer": None, "citations": []},
    {"status": "insufficient_evidence", "answer": "guess", "citations": []},
    {"status": "insufficient_evidence", "answer": None, "citations": ["x"]},
])
def test_inconsistent_outcomes_rejected(payload):
    with pytest.raises(ValidationError):
        GeneratedAnswer.model_validate(payload)


def test_support_and_hop_coverage_require_matching_text_and_identity(case):
    chunks = evidence(case)
    partial = evidence_metrics(case, [chunks[1], chunks[3]])
    assert partial["supporting_recall"] == .5 and partial["supporting_precision"] == .5
    assert partial["hop_coverage"] == .5 and not partial["all_hops_covered"]
    complete = evidence_metrics(case, chunks, [chunks[1].chunk_id, chunks[2].chunk_id])
    assert complete["all_supporting_covered"] and complete["citation_gold_support_precision"] == 1
    altered = chunks[1].model_copy(update={"text": "unrelated"})
    assert evidence_metrics(case, [altered])["supporting_recall"] == 0


def test_gold_answer_aliases_and_future_answerability(source, case):
    assert gold_metrics(case, response(case, answer="Boston, Massachusetts"))["answer_correct"]
    assert not gold_metrics(case, response(case, status="failed"))["answer_correct"]
    negative = load_cases(source, start_record=3, records=1)[0]
    assert gold_metrics(negative, response(negative, status="insufficient_evidence"))["answer_correct"]
    assert not gold_metrics(negative, response(negative, status="failed"))["answer_correct"]


def test_context_budget_selects_whole_ranked_prefix(case):
    chunks = evidence(case)
    assert select_context(chunks, EvaluationSettings(max_context_tokens=1), len) == []
    assert select_context(chunks, EvaluationSettings(max_context_tokens=100000, max_context_chunks=2), len) == chunks[:2]


def fake_metrics():
    return {name: SimpleNamespace(ascore=AsyncMock(return_value=SimpleNamespace(value=.75, reason=None)))
            for name in ("faithfulness", "answer_relevancy", "context_relevance")}


def test_metric_failures_and_abstention_do_not_disappear(case):
    metrics = fake_metrics()
    metrics["faithfulness"].ascore.side_effect = RuntimeError("judge failed")
    scores = asyncio.run(triad_scores(metrics, response(case), evidence(case)))
    assert scores["faithfulness"]["status"] == "failed" and scores["context_relevance"]["value"] == .75
    abstention = asyncio.run(triad_scores(metrics, response(case, status="insufficient_evidence"), evidence(case)))
    assert abstention["faithfulness"]["status"] == "not_applicable" and abstention["context_relevance"]["status"] == "scored"


def test_preflight_checks_every_selected_paragraph(case, monkeypatch):
    calls = []
    from graphrag_pipeline.ingestion.reconcile import ChunkState
    def classify(connection, request):
        calls.append(request.chunk_id)
        return ChunkState("absent" if request.paragraph_idx == 2 else "complete")
    monkeypatch.setattr("graphrag_pipeline.evaluation.runtime.classify_chunk", classify)
    result = graph_readiness(object(), [case])
    assert not result["ready"] and len(calls) == 4
    assert result["missing_record_ids"] == [case.gold.id]


class FakeRetriever:
    def __init__(self, chunks, fail=None, mismatched=False):
        self.chunks, self.fail, self.mismatched = chunks, fail, mismatched
        self.calls = []

    def embed(self, question):
        return [1., 0.]

    def retrieve(self, backend, vector, record_id):
        self.calls.append((backend, vector, record_id))
        if backend == self.fail:
            raise RuntimeError("retrieval failed")
        return self.chunks[:1] if self.mismatched and backend == "graphrag" else self.chunks


def run_fake(case, output, **kwargs):
    retriever = kwargs.pop("retriever", FakeRetriever(evidence(case)))
    llm = MagicMock()
    llm.invoke.return_value = SimpleNamespace(content=json.dumps({"answer": "Boston", "citations": [evidence(case)[1].chunk_id]}))
    result = asyncio.run(run_evaluation([case], retriever, llm, fake_metrics(),
        EvaluationSettings(max_context_tokens=100000), output, count_tokens=len,
        snapshot=kwargs.pop("snapshot", lambda: {"sha256": "same"}), **kwargs))
    return result, retriever, llm


def test_paired_artifacts_use_existing_graph_and_exact_answer_context(case, tmp_path):
    (rows, summary), retriever, llm = run_fake(case, tmp_path / "run")
    stored = [json.loads(s) for s in (tmp_path / "run/records.jsonl").read_text().splitlines()]
    assert len(rows) == len(stored) == summary["expected_rows"] == 4
    assert len(llm.invoke.call_args_list) == 4
    assert [r[2] for r in retriever.calls] == [case.gold.id, case.gold.id, None, None]
    assert all(r["paired_seeds_consistent"] and r["graph_added_supporting_chunk_ids"] == [] for r in stored)
    for row in stored:
        assert row["answer_context"] in row["answer_prompt"]
        assert row["labels"]["question_decomposition"]
        assert '"paragraph_support_idx"' not in row["answer_prompt"]
    assert summary["groups"]["record/vectorrag"]["answer_accuracy"]["mean"] == 1
    assert summary["snapshot_consistent"]


@pytest.mark.parametrize("fail,mismatch", [("graphrag", False), (None, True)])
def test_failed_retrieval_and_seed_mismatch_retained(case, tmp_path, fail, mismatch):
    (rows, _), _, _ = run_fake(case, tmp_path / "run", retriever=FakeRetriever(evidence(case), fail, mismatch))
    assert any(r["response"]["status"] == "failed" for r in rows)
    assert not any(r["paired_seeds_consistent"] for r in rows)


def test_snapshot_change_marks_comparison_invalid(case, tmp_path):
    snapshots = iter([{"sha256": "before"}, {"sha256": "after"}])
    (_, summary), _, _ = run_fake(case, tmp_path / "run", snapshot=lambda: next(snapshots))
    assert summary["snapshot_consistent"] is False


def test_evaluation_refuses_to_overwrite_artifacts(case, tmp_path):
    (tmp_path / "manifest.json").write_text("user data")
    with pytest.raises(ValueError, match="already contains"):
        run_fake(case, tmp_path)
    assert (tmp_path / "manifest.json").read_text() == "user data"


def test_missing_graph_does_not_trigger_preparation(source, monkeypatch):
    monkeypatch.setattr("graphrag_pipeline.evaluation.command.Neo4jSettings.from_env", lambda: SimpleNamespace())
    monkeypatch.setattr("graphrag_pipeline.evaluation.command.model_settings_from_env", lambda: object())
    monkeypatch.setattr("graphrag_pipeline.evaluation.command.Neo4jConnection", MagicMock())
    monkeypatch.setattr("graphrag_pipeline.evaluation.command.graph_readiness", lambda *a: {"ready": False, "issues": ["missing"]})
    prepare = AsyncMock()
    monkeypatch.setattr("graphrag_pipeline.evaluation.command.load_selected_records", prepare)
    args = build_parser().parse_args(["run", "--dataset-path", str(source), "--records", "1"])
    code, result = asyncio.run(execute(args))
    assert code == 2 and not result["preflight"]["ready"]
    prepare.assert_not_called()


@pytest.mark.parametrize("provider", ["ollama", "openrouter"])
def test_real_ragas_metrics_through_mocked_provider_http(case, monkeypatch, provider):
    """Exercise RAGAS/Instructor/SDK code for both endpoints without paid calls."""
    requests = []
    def handle(request):
        body = json.loads(request.content)
        requests.append((str(request.url), body))
        if request.url.path.endswith("embeddings"):
            items = body["input"] if isinstance(body["input"], list) else [body["input"]]
            return httpx.Response(200, json={"object": "list", "model": "embedding", "data": [
                {"object": "embedding", "index": i, "embedding": [1., 0., 0.]} for i, _ in enumerate(items)], "usage": {"prompt_tokens": 1, "total_tokens": 1}})
        text = json.dumps(body["messages"])
        if "noncommittal" in text:
            content = {"question": case.gold.question, "noncommittal": 0}
        elif "verdict" in text:
            content = {"statements": [{"statement": "Boston is the headquarters.", "reason": "supported", "verdict": 1}]}
        elif "rating" in text:
            content = {"rating": 2}
        else:
            content = {"statements": ["Boston is the headquarters."]}
        return httpx.Response(200, json={"id": "mock", "object": "chat.completion", "created": 0, "model": "judge",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(content)}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})
    clients = []
    class TestClient(AsyncOpenAI):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))
            clients.append(self)
    monkeypatch.setattr("openai.AsyncOpenAI", TestClient)
    async def score():
        settings = EvaluatorSettings(provider, "judge", "embedding", "http://provider.test/v1", "test-key")
        metrics, sdk = build_metrics(settings)
        try:
            return await triad_scores(metrics, response(case), evidence(case))
        finally:
            await sdk.close()
    results = asyncio.run(score())
    assert {name: r["status"] for name, r in results.items()} == {name: "scored" for name in results}
    assert all(r["value"] == pytest.approx(1.) for r in results.values())
    assert all(url.startswith("http://provider.test/v1/") for url, _ in requests)
    chat = [b for url, b in requests if url.endswith("completions")]
    if provider == "openrouter":
        assert all(b["provider"]["allow_fallbacks"] is False for b in chat)


def test_evaluator_settings_independent_and_secrets_excluded(monkeypatch):
    monkeypatch.setenv("GRAPH_RAG_PROVIDER", "ollama")
    monkeypatch.setenv("RAGAS_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    monkeypatch.setenv("RAGAS_LLM_MODEL", "judge")
    settings = EvaluatorSettings.from_env()
    assert settings.provider == "openrouter" and settings.llm_model == "judge"
    assert "secret" not in repr(settings) and "api_key" not in settings.public()


def test_existing_graph_command_runs_without_invoking_ingestion(source, tmp_path, monkeypatch):
    case = load_cases(source, records=1)[0]
    import tiktoken
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: SimpleNamespace(encode=lambda text, **kw: list(text)))
    monkeypatch.setattr("graphrag_pipeline.evaluation.command.Neo4jSettings.from_env", lambda: SimpleNamespace(database="neo4j"))
    from graphrag_pipeline.config import OllamaSettings
    monkeypatch.setattr("graphrag_pipeline.evaluation.command.model_settings_from_env", lambda: OllamaSettings("http://local", "answer", "embed"))
    monkeypatch.setattr("graphrag_pipeline.evaluation.command.Neo4jConnection", MagicMock())
    monkeypatch.setattr("graphrag_pipeline.evaluation.command.graph_readiness", lambda *a: {"ready": True})
    monkeypatch.setattr("graphrag_pipeline.evaluation.command.verify_retrieval_prerequisites", lambda *a, **k: SimpleNamespace(to_dict=lambda: {}))
    monkeypatch.setattr("graphrag_pipeline.evaluation.command.graph_snapshot", lambda *a: {"sha256": "same"})
    monkeypatch.setattr("graphrag_pipeline.evaluation.command.PairedRetriever", lambda *a: FakeRetriever(evidence(case)))
    llm = MagicMock()
    llm.invoke.return_value = SimpleNamespace(content=json.dumps({"answer": "Boston", "citations": [evidence(case)[1].chunk_id]}))
    monkeypatch.setattr("graphrag_pipeline.evaluation.command.build_answer_llm", lambda *a: llm)
    monkeypatch.setattr("graphrag_pipeline.evaluation.command.build_metrics", lambda *a: (fake_metrics(), SimpleNamespace(close=AsyncMock())))
    prepare = AsyncMock()
    monkeypatch.setattr("graphrag_pipeline.evaluation.command.load_selected_records", prepare)
    args = build_parser().parse_args(["run", "--dataset-path", str(source), "--records", "1", "--output", str(tmp_path / "run")])
    code, result = asyncio.run(execute(args))
    assert code == 0 and result["completed_rows"] == 4
    prepare.assert_not_called()


def test_prepare_passes_only_selected_records_to_resumable_loader(source, monkeypatch):
    summary = SimpleNamespace(to_dict=lambda: {"records_succeeded": 1}, chunks_failed=[], dead_lettered_records=[], chunks_needs_investigation=[], records_succeeded=1)
    prepare = AsyncMock(return_value=summary)
    monkeypatch.setattr("graphrag_pipeline.evaluation.command.load_selected_records", prepare)
    args = build_parser().parse_args(["prepare", "--dataset-path", str(source), "--start-record", "2", "--records", "1"])
    code, _ = asyncio.run(execute(args))
    assert code == 0
    requests = prepare.call_args.args[0]
    assert [r.record_id for r in requests] == ["fixture_valid_alias_002"]


def test_complete_retrieval_and_selected_graph_context_stay_distinct(case, tmp_path):
    class Expanded(FakeRetriever):
        def retrieve(self, backend, vector, record_id):
            chunks = super().retrieve(backend, vector, record_id)
            return [e.model_copy(update={"graph_nodes": [GraphNode(node_id="n", name="x" * 10000)]}) for e in chunks] if backend == "graphrag" else chunks
    llm = MagicMock()
    llm.invoke.return_value = SimpleNamespace(content=json.dumps({"answer": "Boston", "citations": [evidence(case)[1].chunk_id]}))
    output = tmp_path / "run"
    asyncio.run(run_evaluation([case], Expanded(evidence(case)), llm, fake_metrics(),
        EvaluationSettings(max_context_tokens=2000, conditions=("record",)), output, count_tokens=len))
    rows = [json.loads(r) for r in (output / "records.jsonl").read_text().splitlines()]
    graph = rows[1]
    assert len(graph["retrieval_evidence"]) == 4 and graph["answer_evidence"] == []
    assert graph["context_selected"] and graph["graph_component_complete"]
    assert graph["response"]["status"] == "insufficient_evidence"
    assert graph["retrieval_context_relevance"]["status"] == "scored"
    assert graph["ragas"]["faithfulness"]["status"] == "not_applicable"


def test_metric_timeout_is_a_recorded_error(case):
    async def slow(**kwargs):
        await asyncio.sleep(1)
    metrics = fake_metrics()
    metrics["faithfulness"].ascore = slow
    scores = asyncio.run(triad_scores(metrics, response(case), evidence(case), timeout=.001))
    assert scores["faithfulness"]["status"] == "failed"
    assert scores["answer_relevancy"]["status"] == "scored"


def test_selected_ingestion_scans_source_once(source, monkeypatch):
    from graphrag_pipeline.ingestion.load_musique_dataset import collect_selected_requests
    cases = load_cases(source)
    read_count = 0
    original = Path.open
    def counted(self, *args, **kwargs):
        nonlocal read_count
        if self == source and args and args[0] == "r":
            read_count += 1
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", counted)
    requests, failures, count = collect_selected_requests([c.query(str(i)) for i, c in enumerate(cases)])
    assert not failures and count == 3 and len(requests) == 9
    assert read_count == 1
