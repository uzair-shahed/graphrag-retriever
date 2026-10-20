"""Provider wire formats, resume safety and deterministic benchmark scoring."""

import json
import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from graphrag_pipeline.extraction_benchmark.artifacts import (
    atomic_text, load_gold, read_json, read_jsonl, run_lock, write_jsonl,
)
from graphrag_pipeline.extraction_benchmark.command import build_parser, execute, main
from graphrag_pipeline.extraction_benchmark.contracts import ParagraphExtraction, validate_extraction_response
from graphrag_pipeline.extraction_benchmark.providers import (
    Generation, IncompleteGeneration, JSONProvider, ModelSettings, parse_model,
)
from graphrag_pipeline.extraction_benchmark.records import load_paragraphs
from graphrag_pipeline.extraction_benchmark.runner import report, run_extraction
from graphrag_pipeline.extraction_benchmark.scoring import aggregate, normalize, score


@pytest.fixture
def cohort(tmp_path):
    path = tmp_path / "source.jsonl"
    path.write_text(json.dumps({
        "id": "record", "question": "QUESTION_SECRET", "answer": "ANSWER_SECRET",
        "paragraphs": [{"idx": i, "title": "Meeting", "paragraph_text": "😀 Alice met Bob.",
                        "is_supporting": True} for i in range(2)],
    }), encoding="utf-8")
    return load_paragraphs(path, count=2)


def payload(request):
    return {"mentions": [
        {"mention_id": name.lower(), "mention": name, "entity_type": "Person",
         "evidence_span": {"start": request.text.index(name),
                           "end": request.text.index(name) + len(name), "text": name}}
        for name in ("Alice", "Bob")
    ], "triples": [{"source_mention_id": "alice", "target_mention_id": "bob",
                    "relationship_type": "MET", "confidence": 0.9,
                    "evidence_span": {"start": 0, "end": len(request.text), "text": request.text}}]}


def content(request):
    return json.dumps(payload(request))


def settings(provider="openrouter", **kwargs):
    return ModelSettings(provider, "gold" if provider == "openrouter" else "qwen:4b",
                         "http://provider.test/api/v1" if provider == "openrouter" else "http://provider.test",
                         api_key="secret-key" if provider == "openrouter" else "",
                         retries=0, **kwargs)


class FakeFactory:
    def __init__(self, behavior=None):
        self.calls = []
        self.checks = []
        self.behavior = behavior

    def __call__(self, model):
        def generate(request):
            self.calls.append((model.spec, request.chunk_id))
            output = self.behavior(model, request) if self.behavior else content(request)
            return Generation(output, {"usage": {"completion_tokens": 10}, "finish_reason": "stop"})
        factory = self
        class Client:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def check(self):
                factory.checks.append(model.spec)
                return {"mock": True}
        client = Client()
        client.generate = generate
        return client


def gold_run(cohort, output, factory=None, **kwargs):
    return run_extraction(cohort, [settings(reasoning_effort="high")], output,
                          kind="gold", provider_factory=factory or FakeFactory(),
                          show_progress=False, **kwargs)


@pytest.mark.parametrize("provider", ["openrouter", "ollama"])
def test_provider_http_schema_reasoning_and_label_isolation(cohort, provider):
    seen = []
    def handle(request):
        body = json.loads(request.content) if request.content else None
        seen.append((request.url.path, body))
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "gold",
                "supported_parameters": ["structured_outputs", "reasoning"],
                "reasoning": {"supported_efforts": ["high", "low"]}}]})
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["completion", "thinking"]})
        if provider == "openrouter":
            return httpx.Response(200, json={"id": "completion", "model": "gold", "usage": {"completion_tokens": 10},
                "choices": [{"finish_reason": "stop", "message": {"content": content(cohort[0])}}]})
        return httpx.Response(200, json={"model": "qwen:4b", "done": True, "done_reason": "stop",
            "message": {"content": content(cohort[0])}, "eval_count": 10})
    model = settings(provider, reasoning_effort="high") if provider == "openrouter" else settings(provider, think=False)
    with JSONProvider(model, transport=httpx.MockTransport(handle)) as client:
        client.check()
        generated = client.generate(cohort[0])
    assert validate_extraction_response(cohort[0], generated.content).triples[0].relationship_type == "MET"
    body = seen[-1][1]
    assert json.loads(body["messages"][1]["content"]) == {"title": "Meeting", "text": cohort[0].text}
    assert "QUESTION_SECRET" not in json.dumps(body) and "ANSWER_SECRET" not in json.dumps(body)
    assert "record_id" not in json.dumps(body) and "is_supporting" not in json.dumps(body)
    assert body["stream"] is False
    if provider == "openrouter":
        assert body["response_format"]["json_schema"]["schema"] == ParagraphExtraction.model_json_schema()
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["strict"] is True
        assert body["reasoning"] == {"effort": "high", "exclude": True}
        assert body["provider"] == {"require_parameters": True, "allow_fallbacks": False}
        assert "temperature" not in body
    else:
        assert body["format"] == ParagraphExtraction.model_json_schema()
        assert body["think"] is False
        assert body["options"]["num_predict"] == model.max_tokens


@pytest.mark.parametrize("reasoning, supported, message", [
    ({"supported_efforts": ["low"]}, ["reasoning", "response_format"], "does not support reasoning effort"),
    (None, ["response_format"], "does not advertise reasoning"),
    ({"supported_efforts": ["high"]}, ["reasoning"], "structured output"),
])
def test_unsupported_gold_config_is_rejected_before_generation(reasoning, supported, message):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"data": [
        {"id": "gold", "supported_parameters": supported, "reasoning": reasoning}]}))
    with JSONProvider(settings(reasoning_effort="high"), transport=transport) as client:
        with pytest.raises(ValueError, match=message):
            client.check()


@pytest.mark.parametrize("provider", ["openrouter", "ollama"])
def test_truncated_even_parseable_response_is_failure(cohort, provider):
    response = {"choices": [{"finish_reason": "length", "message": {"content": content(cohort[0])}}]} if provider == "openrouter" else {
        "done": True, "done_reason": "length", "message": {"content": content(cohort[0])}}
    with JSONProvider(settings(provider), transport=httpx.MockTransport(lambda request: httpx.Response(200, json=response))) as client:
        with pytest.raises(IncompleteGeneration) as raised:
            client.generate(cohort[0])
    assert raised.value.generation.content == content(cohort[0])


@pytest.mark.parametrize("status, expected", [(429, 2), (503, 2), (400, 1), (401, 1)])
def test_http_retry_bounds(status, expected, monkeypatch):
    requests = []
    monkeypatch.setattr("graphrag_pipeline.extraction_benchmark.providers.time.sleep", lambda seconds: None)
    def handle(request):
        requests.append(request)
        return httpx.Response(status, json={"error": "secret-key"})
    with JSONProvider(replace(settings(), retries=1), transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(ValueError, match=f"HTTP {status}") as raised:
            client.check()
    assert len(requests) == expected
    assert "secret-key" not in str(raised.value)


def test_model_syntax_and_independent_environment(monkeypatch):
    assert parse_model("ollama:qwen3:4b") == ("ollama", "qwen3:4b")
    for invalid in ("qwen3:4b", "ollama:", "openai:model", "openrouter: model"):
        with pytest.raises(ValueError): parse_model(invalid)
    monkeypatch.setenv("GRAPH_RAG_PROVIDER", "invalid")
    monkeypatch.setenv("RAGAS_PROVIDER", "invalid")
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    local = ModelSettings.from_env("ollama:qwen3:4b", ollama_think="true")
    hosted = ModelSettings.from_env("openrouter:vendor/model", reasoning_effort="high")
    assert local.think is True and hosted.reasoning_effort == "high"
    assert "api_key" not in hosted.public() and "secret" not in repr(hosted)
    with pytest.raises(ValueError, match="without credentials"):
        replace(hosted, base_url="https://user:secret@example.test")


def test_scoring_types_direction_normalization_duplicates_and_empty(cohort):
    request = cohort[0]
    gold = validate_extraction_response(request, content(request))
    modified = payload(request)
    modified["mentions"][0]["entity_type"] = "person!"
    modified["triples"][0]["relationship_type"] = "met."
    duplicate = {**modified["mentions"][0], "mention_id": "alice-again"}
    modified["mentions"].append(duplicate)
    modified["triples"].append(dict(modified["triples"][0]))
    prediction = validate_extraction_response(request, json.dumps(modified))
    scores = score(gold, prediction)
    assert scores["entities"]["tp"] == 2 and scores["relations"]["tp"] == 1
    assert scores["entities"]["f1"] == 1 and scores["relations"]["f1"] == 1
    modified["triples"][0]["source_mention_id"] = "bob"
    modified["triples"][0]["target_mention_id"] = "alice"
    modified["triples"] = modified["triples"][:1]
    reversed_scores = score(gold, validate_extraction_response(request, json.dumps(modified)))
    assert reversed_scores["relations"]["tp"] == 0
    assert reversed_scores["relations"]["fp"] == reversed_scores["relations"]["fn"] == 1
    modified["mentions"][0]["entity_type"] = "Organization"
    modified["mentions"] = modified["mentions"][:2]
    changed = score(gold, validate_extraction_response(request, json.dumps(modified)))
    assert changed["entities"]["tp"] == 1 and changed["entities"]["fp"] == changed["entities"]["fn"] == 1
    empty = score(gold, None)
    assert empty["entities"]["fn"] == 2 and empty["relations"]["fn"] == 1
    combined = aggregate([scores, empty])
    assert combined["entities"]["precision"] == 1 and combined["entities"]["recall"] == 0.5
    assert combined["entities"]["f1"] == pytest.approx(2 / 3)
    assert normalize(" Ａlice—SMITH_  Jr. ") == "alice smith jr"
    assert aggregate([])["entities"]["f1"] == 0


def test_gold_interrupt_resume_skips_paid_work_and_offline_report(cohort, tmp_path):
    output = tmp_path / "gold"
    def interrupt(model, request):
        if request.paragraph_idx == 1:
            raise KeyboardInterrupt
        return content(request)
    with pytest.raises(KeyboardInterrupt):
        gold_run(cohort, output, FakeFactory(interrupt))
    assert len(read_jsonl(output / "gold.jsonl")) == 1
    assert not (output / ".benchmark.lock").exists()
    assert report(output)["models"]["openrouter:gold"]["pending"] == 1
    factory = FakeFactory()
    assert gold_run(cohort, output, factory, resume=True)["successful"]
    assert factory.calls == [("openrouter:gold", cohort[1].chunk_id)]
    factory = FakeFactory()
    gold_run(cohort, output, factory, resume=True)
    assert factory.calls == factory.checks == []
    requests, reference, fingerprint = load_gold(output)
    assert requests == cohort and len(reference) == 2 and len(fingerprint) == 64
    assert "secret-key" not in (output / "manifest.json").read_text()


def test_saved_failures_require_explicit_retry_and_incomplete_gold_rejected(cohort, tmp_path):
    output = tmp_path / "gold"
    factory = FakeFactory(lambda model, request: "malformed" if request.paragraph_idx else content(request))
    assert not gold_run(cohort, output, factory)["successful"]
    with pytest.raises(ValueError, match="incomplete or failed"):
        load_gold(output)
    no_work = FakeFactory()
    gold_run(cohort, output, no_work, resume=True)
    assert no_work.calls == []
    retry = FakeFactory()
    assert gold_run(cohort, output, retry, resume=True, retry_failed=True)["successful"]
    assert retry.calls == [("openrouter:gold", cohort[1].chunk_id)]
    assert [row["attempts"] for row in read_jsonl(output / "gold.jsonl")] == [1, 2]


def test_candidate_matrix_scores_failures_and_needs_no_original_dataset(cohort, tmp_path):
    gold = tmp_path / "gold"
    gold_run(cohort, gold)
    requests, reference, fingerprint = load_gold(gold)
    Path(cohort[0].provenance.source_file).unlink()
    candidates = [settings(), settings("ollama")]
    factory = FakeFactory(lambda model, request: "not JSON" if model.provider == "ollama" and request.paragraph_idx else content(request))
    output = tmp_path / "candidates"
    summary = run_extraction(requests, candidates, output, kind="candidate", reference=reference,
                             gold_fingerprint=fingerprint, provider_factory=factory, show_progress=False)
    assert len(factory.calls) == 4 and summary["complete"] and not summary["successful"]
    local = summary["models"]["ollama:qwen:4b"]
    assert local["failed"] == 1 and local["micro"]["entities"]["recall"] == 0.5
    assert summary["models"]["openrouter:gold"]["micro"]["entities"]["f1"] == 1
    saved = read_jsonl(output / "predictions.jsonl")
    assert saved[-1]["status"] == "failed" and saved[-1]["result"] is None
    assert saved[-1]["scores"]["entities"]["fn"] == 2
    # Reporting is self-contained even if the original gold directory disappears.
    for file in gold.iterdir(): file.unlink()
    gold.rmdir()
    assert report(output) == summary


@pytest.mark.parametrize("change", ["settings", "source", "duplicate_row", "corrupt_raw"])
def test_resume_rejects_incompatible_or_corrupted_artifacts(cohort, tmp_path, change):
    output = tmp_path / "gold"
    gold_run(cohort, output)
    factory = FakeFactory()
    models = [settings(reasoning_effort="high")]
    if change == "settings": models = [replace(models[0], max_tokens=12)]
    elif change == "source": cohort = [cohort[0].model_copy(update={"text": "different"}), cohort[1]]
    else:
        rows = read_jsonl(output / "gold.jsonl")
        if change == "duplicate_row": rows.append(rows[0])
        else: rows[0]["raw_content"] = "not JSON"
        write_jsonl(output / "gold.jsonl", rows)
    with pytest.raises(ValueError):
        run_extraction(cohort, models, output, kind="gold", resume=True,
                       provider_factory=factory, show_progress=False)
    assert factory.calls == factory.checks == []


def test_output_locks_and_atomic_replacement_preserve_saved_state(tmp_path, monkeypatch):
    output = tmp_path / "locked"
    with run_lock(output):
        with pytest.raises(ValueError, match="locked"):
            with run_lock(output): pass
    target = tmp_path / "atomic.txt"
    target.write_text("old")
    def fail(*args): raise OSError("disk error")
    monkeypatch.setattr("graphrag_pipeline.extraction_benchmark.artifacts.os.replace", fail)
    with pytest.raises(OSError): atomic_text(target, "new")
    assert target.read_text() == "old"


@pytest.mark.parametrize("corruption", ["reference", "scores", "cohort"])
def test_offline_report_rejects_changed_candidate_evidence(cohort, tmp_path, corruption):
    gold = tmp_path / "gold"
    gold_run(cohort, gold)
    requests, reference, fingerprint = load_gold(gold)
    output = tmp_path / "candidates"
    run_extraction(requests, [settings("ollama")], output, kind="candidate", reference=reference,
                   gold_fingerprint=fingerprint, provider_factory=FakeFactory(), show_progress=False)
    name = {"reference": "reference.jsonl", "scores": "predictions.jsonl", "cohort": "cohort.jsonl"}[corruption]
    path = output / name
    rows = read_jsonl(path)
    if corruption == "reference": rows[0]["mentions"][0]["entity_type"] = "Organization"
    elif corruption == "scores": rows[0]["scores"]["entities"]["tp"] += 1
    else: rows[0]["text"] = "changed"
    write_jsonl(path, rows)
    with pytest.raises(ValueError): report(output)


def test_cli_dry_runs_require_no_services_or_artifacts(cohort, tmp_path, monkeypatch):
    source = cohort[0].provenance.source_file
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    parser = build_parser()
    output = tmp_path / "not-created"
    args = parser.parse_args(["gold", "--dataset-path", source, "--paragraphs", "2", "--model", "vendor/model",
                              "--reasoning-effort", "high", "--output", str(output), "--dry-run"])
    code, preview = execute(args)
    assert code == 0 and preview["models"][0]["model"] == "vendor/model"
    assert not output.exists()
    args = parser.parse_args(["select", "--dataset-path", source, "--paragraphs", "1"])
    assert execute(args)[1]["paragraphs"] == 1
    with pytest.raises(SystemExit):
        parser.parse_args(["gold", "--model", "vendor/model", "--output", str(output)])


def test_cli_gold_candidate_and_report_end_to_end(cohort, tmp_path, monkeypatch):
    import graphrag_pipeline.extraction_benchmark.command as command
    original = run_extraction
    factory = FakeFactory()
    monkeypatch.setattr(command, "run_extraction", lambda *args, **kwargs: original(*args, **kwargs, provider_factory=factory))
    parser = build_parser()
    gold = tmp_path / "gold"
    code, summary = execute(parser.parse_args([
        "gold", "--dataset-path", cohort[0].provenance.source_file, "--paragraphs", "2",
        "--model", "gold", "--reasoning-effort", "high", "--output", str(gold), "--no-progress",
    ]))
    assert code == 0 and summary["successful"]
    output = tmp_path / "results"
    args = parser.parse_args(["run", "--gold", str(gold), "--model", "ollama:qwen:4b",
                             "--model", "openrouter:candidate", "--output", str(output), "--no-progress"])
    assert execute(args)[0] == 0
    args = parser.parse_args(["report", "--input", str(output)])
    assert execute(args)[1]["models"]["ollama:qwen:4b"]["micro"]["relations"]["f1"] == 1


def test_cli_invalid_settings_are_json_errors_and_nonzero(cohort, tmp_path, monkeypatch, capsys):
    import sys
    monkeypatch.setattr(sys, "argv", ["benchmark", "gold", "--model", "gold", "--reasoning-effort", "high",
        "--dataset-path", cohort[0].provenance.source_file, "--paragraphs", "2", "--max-tokens", "0",
        "--output", str(tmp_path / "unused")])
    with pytest.raises(SystemExit) as raised: main()
    assert raised.value.code == 2
    assert json.loads(capsys.readouterr().out)["status"] == "failed"


@pytest.mark.parametrize("answer_provider", ["ollama", "openrouter"])
@pytest.mark.parametrize("eval_provider", ["ollama", "openrouter"])
def test_existing_answer_and_ragas_provider_roles_remain_independent(tmp_path, monkeypatch, answer_provider, eval_provider):
    """Exercise the existing evaluation command's four provider combinations."""
    from graphrag_pipeline.config import OllamaSettings, OpenRouterSettings
    from graphrag_pipeline.evaluation import command
    from graphrag_pipeline.evaluation.providers import EvaluatorSettings

    local = OllamaSettings("http://local", "local-answer", "local-embedding")
    hosted = OpenRouterSettings("secret", "https://hosted.test/v1", "hosted-answer", "hosted-embedding", 3, 100, True, False)
    case = SimpleNamespace(source_sha256="sha", source_line=1, gold=SimpleNamespace(id="record"))
    monkeypatch.setattr(command, "selection_from_args", lambda args: [case])
    monkeypatch.setattr(command.Neo4jSettings, "from_env", lambda: SimpleNamespace(database="neo4j"))
    monkeypatch.setattr(command, "model_settings_from_env", lambda: local)
    monkeypatch.setattr(command.OllamaSettings, "from_env", lambda: local)
    monkeypatch.setattr(command.OpenRouterSettings, "from_env", lambda: hosted)
    monkeypatch.setattr(command, "Neo4jConnection", MagicMock())
    monkeypatch.setattr(command, "graph_readiness", lambda *args: {"ready": True})
    monkeypatch.setattr(command, "verify_retrieval_prerequisites", lambda *args, **kwargs: SimpleNamespace(to_dict=lambda: {}))
    monkeypatch.setattr(command, "PairedRetriever", lambda *args: object())
    selected_answers, selected_evaluators = [], []
    monkeypatch.setattr(command, "build_answer_llm", lambda model: selected_answers.append(model))
    monkeypatch.setattr(command.EvaluatorSettings, "from_env", lambda provider: EvaluatorSettings(provider, "judge", "embed", "https://judge.test/v1", "key"))
    monkeypatch.setattr(command, "build_metrics", lambda evaluator: (selected_evaluators.append(evaluator), SimpleNamespace(close=AsyncMock())))
    runner = AsyncMock(return_value=([], {"snapshot_consistent": True}))
    monkeypatch.setattr(command, "run_evaluation", runner)
    args = command.build_parser().parse_args(["run", "--answer-provider", answer_provider,
        "--eval-provider", eval_provider, "--output", str(tmp_path / "evaluation"), "--no-progress"])
    assert asyncio.run(command.execute(args))[0] == 0
    assert selected_answers == [local if answer_provider == "ollama" else hosted]
    assert selected_evaluators[0].provider == eval_provider
    assert runner.call_args.kwargs["configuration"]["retrieval_models"]["llm_model"] == "local-answer"


def test_all_readme_and_reference_commands_parse_and_document_every_flag():
    import re
    import shlex
    parser = build_parser()
    options = set()
    for action in parser._actions:
        options.update(action.option_strings)
        if hasattr(action, "choices") and isinstance(action.choices, dict):
            for subparser in action.choices.values():
                options.update(option for item in subparser._actions for option in item.option_strings)
    for path in (Path("README.md"), Path("agent-coordination/entity-extraction-benchmark/entity-extraction-benchmark.md")):
        text = path.read_text()
        documented = set(re.findall(r"--[a-z][a-z-]+", text))
        assert {option for option in options if option.startswith("--")} <= documented
        commands = re.findall(r"^uv run --locked graphrag-extraction-benchmark .+$", text, re.MULTILINE)
        assert commands
        for command in commands:
            arguments = shlex.split(command)[4:]
            if "--help" in arguments:
                with pytest.raises(SystemExit) as raised: parser.parse_args(arguments)
                assert raised.value.code == 0
            else:
                parser.parse_args(arguments)
