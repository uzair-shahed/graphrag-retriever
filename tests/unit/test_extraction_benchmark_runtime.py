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
    APPEND_TRANSACTION, atomic_text, load_gold, load_run, read_json, read_jsonl, run_lock, write_jsonl,
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
        {"mention_id": name.lower(), "mention": name, "entity_type": "Person"}
        for name in ("Alice", "Bob")
    ], "triples": [{"source_mention_id": "alice", "target_mention_id": "bob",
                    "relationship_type": "MET"}]}


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
    requests, reference, fingerprint, skipped = load_gold(output)
    assert skipped == []
    assert requests == cohort and len(reference) == 2 and len(fingerprint) == 64
    assert "secret-key" not in (output / "manifest.json").read_text()


def test_saved_failures_require_explicit_retry_and_partial_gold_is_usable(cohort, tmp_path):
    output = tmp_path / "gold"
    factory = FakeFactory(lambda model, request: "malformed" if request.paragraph_idx else content(request))
    assert not gold_run(cohort, output, factory)["successful"]
    requests, reference, _, skipped = load_gold(output)
    assert requests == cohort and list(reference) == [cohort[0].chunk_id]
    assert skipped == [{"chunk_id": cohort[1].chunk_id, "reason": "gold_failed", "error_type": "ValidationError"}]
    no_work = FakeFactory()
    gold_run(cohort, output, no_work, resume=True)
    assert no_work.calls == []
    retry = FakeFactory()
    assert gold_run(cohort, output, retry, resume=True, retry_failed=True)["successful"]
    assert retry.calls == [("openrouter:gold", cohort[1].chunk_id)]
    assert [row["attempts"] for row in read_jsonl(output / "gold.jsonl")] == [1, 2]


def test_cli_retry_reports_count_mismatch_then_retries_only_failures(cohort, tmp_path, monkeypatch):
    import graphrag_pipeline.extraction_benchmark.command as command

    factory = FakeFactory(lambda model, request: "malformed" if request.paragraph_idx else content(request))
    monkeypatch.setattr(command, "run_extraction", lambda *args, **kwargs: run_extraction(
        *args, **kwargs, provider_factory=factory))
    parser = build_parser()
    output = tmp_path / "gold"
    arguments = ["gold", "--dataset-path", cohort[0].provenance.source_file,
                 "--model", "gold", "--reasoning-effort", "high", "--output", str(output),
                 "--no-progress"]
    assert execute(parser.parse_args([*arguments, "--paragraphs", "2"]))[0] == 1
    snapshots = {path.name: path.read_bytes() for path in output.iterdir()}
    factory.calls.clear()
    factory.checks.clear()
    with pytest.raises(ValueError, match=r"saved run has 2 paragraphs, but requested 1.*--paragraphs 2"):
        execute(parser.parse_args([*arguments, "--paragraphs", "1", "--resume", "--retry-failed"]))
    assert factory.calls == factory.checks == []
    assert snapshots == {path.name: path.read_bytes() for path in output.iterdir()}

    factory.behavior = None
    code, summary = execute(parser.parse_args([
        *arguments, "--paragraphs", "2", "--resume", "--retry-failed"]))
    assert code == 0 and summary["successful"]
    assert factory.calls == [("openrouter:gold", cohort[1].chunk_id)]
    saved = read_jsonl(output / "gold.jsonl")
    original = [json.loads(line) for line in snapshots["gold.jsonl"].decode().splitlines()]
    assert saved[0] == original[0]
    assert saved[1]["attempts"] == 2


def test_candidate_matrix_scores_failures_and_needs_no_original_dataset(cohort, tmp_path):
    gold = tmp_path / "gold"
    gold_run(cohort, gold)
    requests, reference, fingerprint, _ = load_gold(gold)
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
    requests, reference, fingerprint, _ = load_gold(gold)
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


def test_cli_unicode_preview_supports_windows_redirected_stdout(cohort, monkeypatch):
    import io
    import sys

    source = Path(cohort[0].provenance.source_file)
    record = json.loads(source.read_text(encoding="utf-8"))
    record["paragraphs"][0]["title"] = "Simi\u0107 \U0001f600"
    source.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    encoded = io.BytesIO()
    stdout = io.TextIOWrapper(encoded, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "argv", ["benchmark", "select", "--paragraphs", "1",
                                     "--dataset-path", cohort[0].provenance.source_file])
    with pytest.raises(SystemExit) as raised:
        main()
    assert raised.value.code == 0
    stdout.flush()
    preview = json.loads(encoded.getvalue().decode("cp1252"))
    assert preview["selection"][0]["title"] == record["paragraphs"][0]["title"]


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


@pytest.fixture
def append_cli(tmp_path, monkeypatch):
    """A real 180-paragraph source spanning records, with mocked providers only."""
    import graphrag_pipeline.extraction_benchmark.command as command

    source = tmp_path / "source.jsonl"
    source.write_text("\n".join(json.dumps({
        "id": name, "paragraphs": [
            {"idx": i, "title": name, "paragraph_text": "Alice met Bob."}
            for i in reversed(range(count))],
    }) for name, count in (("first", 120), ("second", 60))) + "\n", encoding="utf-8")
    factory = FakeFactory()
    monkeypatch.setattr(command, "run_extraction", lambda *args, **kwargs: run_extraction(
        *args, **kwargs, provider_factory=factory))
    output = tmp_path / "gold"
    arguments = ["gold", "--dataset-path", str(source), "--model", "gold",
                 "--reasoning-effort", "high", "--output", str(output), "--no-progress"]

    def invoke(*extra):
        return execute(build_parser().parse_args([*arguments, *extra]))
    return source, output, factory, invoke


def snapshots(directory):
    return {path.name: path.read_bytes() for path in directory.iterdir()}


@pytest.mark.parametrize("selector", [
    ["--additional-paragraphs", "30"],
    ["--start-paragraph", "151", "--end-paragraph", "180"],
])
def test_cli_append_150_to_180_reuses_every_saved_extraction(append_cli, selector):
    source, output, factory, invoke = append_cli
    assert invoke("--paragraphs", "150")[0] == 0
    before = snapshots(output)
    original = read_jsonl(output / "gold.jsonl")
    factory.calls.clear()
    factory.checks.clear()
    code, preview = invoke(*selector, "--dry-run")
    assert code == 0 and preview["paragraphs"] == 180
    assert preview["append"] == {"previous_paragraphs": 150, "additional_paragraphs": 30,
                                 "start_paragraph": 151, "end_paragraph": 180}
    assert snapshots(output) == before and factory.calls == factory.checks == []

    code, summary = invoke(*selector)
    assert code == 0 and summary["successful"]
    expected = load_paragraphs(source, count=180)
    assert factory.calls == [("openrouter:gold", r.chunk_id) for r in expected[150:]]
    assert read_jsonl(output / "gold.jsonl")[:150] == original
    assert all(row["attempts"] == 1 for row in read_jsonl(output / "gold.jsonl"))
    assert load_gold(output)[0] == expected
    manifest = read_json(output / "manifest.json")
    previous = json.loads(before["manifest.json"])
    assert manifest["run_id"] == previous["run_id"] and manifest["created_at"] == previous["created_at"]
    assert summary["models"]["openrouter:gold"]["expected"] == 180
    assert {"output": str(output), **report(output)} == summary
    assert not (output / APPEND_TRANSACTION).exists()


@pytest.mark.parametrize("selector, message", [
    (["--start-paragraph", "150", "--end-paragraph", "180"], "start at paragraph 151"),
    (["--start-paragraph", "152", "--end-paragraph", "180"], "start at paragraph 151"),
    (["--start-paragraph", "151", "--end-paragraph", "150"], "end >= start"),
    (["--start-paragraph", "0", "--end-paragraph", "180"], "positive"),
    (["--start-paragraph", "151", "--end-paragraph", "181"], "contains only 180"),
    (["--start-paragraph", "151"], "requires positive"),
    (["--end-paragraph", "180"], "requires positive"),
    (["--additional-paragraphs", "0"], "must be positive"),
    (["--additional-paragraphs", "-1"], "must be positive"),
    (["--additional-paragraphs", "30", "--end-paragraph", "180"], "cannot be combined"),
    (["--additional-paragraphs", "30", "--paragraphs", "180"], "cannot be combined"),
])
def test_invalid_append_selection_changes_nothing(append_cli, selector, message):
    _, output, factory, invoke = append_cli
    invoke("--paragraphs", "150")
    before = snapshots(output)
    factory.calls.clear()
    factory.checks.clear()
    with pytest.raises(ValueError, match=message):
        invoke(*selector)
    assert snapshots(output) == before and factory.calls == factory.checks == []


def test_append_requires_existing_output_and_mutually_exclusive_selectors(append_cli):
    _, output, _, invoke = append_cli
    with pytest.raises(ValueError, match="existing benchmark run"):
        invoke("--additional-paragraphs", "30")
    assert not output.exists()
    with pytest.raises(SystemExit):
        invoke("--additional-paragraphs", "30", "--start-paragraph", "151", "--end-paragraph", "180")


@pytest.mark.parametrize("change", ["settings", "source"])
@pytest.mark.parametrize("dry_run", [False, True])
def test_append_checks_source_and_settings_before_writes_or_calls(append_cli, change, dry_run):
    source, output, factory, invoke = append_cli
    invoke("--paragraphs", "150")
    before = snapshots(output)
    extra = ["--additional-paragraphs", "30"]
    if change == "settings":
        extra.extend(["--max-tokens", "12"])
    else:
        # Even an unchanged selected prefix cannot be mixed with a new source hash.
        source.write_bytes(source.read_bytes() + b"\n")
    if dry_run:
        extra.append("--dry-run")
    factory.calls.clear()
    factory.checks.clear()
    with pytest.raises(ValueError, match="configuration/source mismatch"):
        invoke(*extra)
    assert snapshots(output) == before and factory.calls == factory.checks == []


@pytest.mark.parametrize("retry", [False, True])
def test_append_preserves_failures_unless_explicitly_retried(append_cli, retry):
    source, output, factory, invoke = append_cli
    first = load_paragraphs(source, count=1)[0].chunk_id
    factory.behavior = lambda model, request: "malformed" if request.chunk_id == first else content(request)
    invoke("--paragraphs", "150")
    original = read_jsonl(output / "gold.jsonl")
    factory.behavior = None
    factory.calls.clear()
    code, summary = invoke("--additional-paragraphs", "30", *(["--resume", "--retry-failed"] if retry else []))
    assert code == (0 if retry else 1) and summary["complete"]
    saved = read_jsonl(output / "gold.jsonl")
    assert saved[1:150] == original[1:]
    assert len(factory.calls) == (31 if retry else 30)
    if retry:
        assert saved[0]["status"] == "succeeded" and saved[0]["attempts"] == 2
    else:
        assert saved[0] == original[0]
    with pytest.raises(ValueError, match="requires --resume"):
        invoke("--additional-paragraphs", "1", "--retry-failed")


def test_interrupted_append_resumes_the_expanded_cohort(append_cli):
    source, output, factory, invoke = append_cli
    invoke("--paragraphs", "150")
    expected = load_paragraphs(source, count=180)
    def interrupt(model, request):
        if request == expected[151]:
            raise KeyboardInterrupt
        return content(request)
    factory.behavior = interrupt
    with pytest.raises(KeyboardInterrupt):
        invoke("--additional-paragraphs", "30")
    before = read_jsonl(output / "gold.jsonl")
    assert len(before) == 151
    factory.calls.clear()
    factory.checks.clear()
    with pytest.raises(ValueError, match="pending paragraphs"):
        invoke("--additional-paragraphs", "1")
    assert factory.calls == factory.checks == []
    factory.behavior = None
    assert invoke("--paragraphs", "180", "--resume")[0] == 0
    assert factory.calls == [("openrouter:gold", r.chunk_id) for r in expected[151:]]
    assert read_jsonl(output / "gold.jsonl")[:151] == before


@pytest.mark.parametrize("selector", [
    ["--additional-paragraphs", "30"],
    ["--start-paragraph", "151", "--end-paragraph", "180"],
])
def test_candidate_append_extends_frozen_gold_and_all_models(append_cli, tmp_path, selector):
    source, gold, factory, invoke = append_cli
    invoke("--paragraphs", "150")
    output = tmp_path / "candidate"
    arguments = ["run", "--gold", str(gold), "--output", str(output),
                 "--model", "ollama:qwen:4b", "--model", "openrouter:candidate", "--no-progress"]
    parser = build_parser()
    assert execute(parser.parse_args(arguments))[0] == 0
    original = read_jsonl(output / "predictions.jsonl")
    frozen = read_jsonl(output / "reference.jsonl")
    before = snapshots(output)
    factory.calls.clear()
    factory.checks.clear()
    with pytest.raises(ValueError, match="gold cohort contains only 150"):
        execute(parser.parse_args([*arguments, *selector]))
    assert snapshots(output) == before and factory.calls == factory.checks == []
    invoke("--additional-paragraphs", "30")
    source.unlink()
    factory.calls.clear()
    code, summary = execute(parser.parse_args([*arguments, *selector]))
    assert code == 0 and len(factory.calls) == 60
    assert read_jsonl(output / "predictions.jsonl")[:300] == original
    assert read_jsonl(output / "reference.jsonl")[:150] == frozen
    assert all(model["expected"] == 180 and model["micro"]["entities"]["f1"] == 1
               for model in summary["models"].values())
    assert {"output": str(output), **report(output)} == summary
    factory.calls.clear()
    factory.checks.clear()
    assert execute(parser.parse_args([*arguments, "--resume"]))[0] == 0
    assert factory.calls == factory.checks == []


def test_candidate_append_can_use_a_prefix_of_larger_gold(append_cli, tmp_path):
    source, gold, factory, invoke = append_cli
    invoke("--paragraphs", "150")
    output = tmp_path / "candidate"
    arguments = ["run", "--gold", str(gold), "--output", str(output), "--model", "ollama:qwen:4b", "--no-progress"]
    parser = build_parser()
    execute(parser.parse_args(arguments))
    invoke("--additional-paragraphs", "30")
    expected = load_paragraphs(source, count=180)
    def interrupt(model, request):
        if request == expected[151]:
            raise KeyboardInterrupt
        return content(request)
    factory.behavior = interrupt
    with pytest.raises(KeyboardInterrupt):
        execute(parser.parse_args([*arguments, "--additional-paragraphs", "10"]))
    factory.behavior = None
    factory.calls.clear()
    assert execute(parser.parse_args([*arguments, "--resume"]))[0] == 0
    assert factory.calls == [("ollama:qwen:4b", r.chunk_id) for r in expected[151:160]]
    assert read_json(output / "manifest.json")["configuration"]["paragraphs"] == 160


def test_candidate_append_rejects_changed_existing_gold(cohort, tmp_path):
    gold = tmp_path / "gold"
    gold_run(cohort[:1], gold)
    requests, reference, fingerprint, _ = load_gold(gold)
    output = tmp_path / "candidate"
    models = [settings("ollama")]
    run_extraction(requests, models, output, kind="candidate", reference=reference,
                   gold_fingerprint=fingerprint, provider_factory=FakeFactory(), show_progress=False)
    gold_run(cohort, gold, append_start=2)
    rows = read_jsonl(gold / "gold.jsonl")
    changed = payload(cohort[0])
    changed["mentions"][0]["entity_type"] = "Organization"
    rows[0]["raw_content"] = json.dumps(changed)
    rows[0]["result"] = validate_extraction_response(cohort[0], rows[0]["raw_content"]).model_dump(mode="json")
    write_jsonl(gold / "gold.jsonl", rows)
    requests, reference, fingerprint, _ = load_gold(gold)
    factory = FakeFactory()
    before = snapshots(output)
    with pytest.raises(ValueError, match="existing gold labels must be unchanged"):
        run_extraction(requests, models, output, kind="candidate", reference=reference,
                       gold_fingerprint=fingerprint, append_start=2, provider_factory=factory)
    assert snapshots(output) == before and factory.calls == factory.checks == []


@pytest.mark.parametrize("target, after_replace, use_report", [
    ("cohort.jsonl", False, False), ("reference.jsonl", False, True),
    ("manifest.json", False, False), ("manifest.json", True, True),
])
@pytest.mark.parametrize("skip_new", [False, True])
def test_append_metadata_commit_recovers_at_each_write_boundary(cohort, tmp_path, monkeypatch, target, after_replace, use_report, skip_new):
    import graphrag_pipeline.extraction_benchmark.artifacts as artifacts

    output = tmp_path / "candidate"
    models = [settings("ollama")]
    reference = {r.chunk_id: validate_extraction_response(r, content(r)) for r in cohort}
    run_extraction(cohort[:1], models, output, kind="candidate", reference=reference,
                   gold_fingerprint="old", provider_factory=FakeFactory(), show_progress=False)
    original = read_jsonl(output / "predictions.jsonl")
    skipped = []
    if skip_new:
        del reference[cohort[1].chunk_id]
        skipped = [{"chunk_id": cohort[1].chunk_id, "reason": "gold_failed", "error_type": "ValidationError"}]
    replace_file = artifacts.os.replace
    def fail(source, destination):
        if Path(destination).name == target:
            if after_replace:
                replace_file(source, destination)
            raise OSError("interrupted append commit")
        return replace_file(source, destination)
    factory = FakeFactory()
    with monkeypatch.context() as patch:
        patch.setattr(artifacts.os, "replace", fail)
        with pytest.raises(OSError, match="interrupted append commit"):
            run_extraction(cohort, models, output, kind="candidate", reference=reference,
                           gold_fingerprint="new", skipped_gold=skipped,
                           append_start=2, provider_factory=factory, show_progress=False)
    assert (output / APPEND_TRANSACTION).exists()
    assert factory.calls == factory.checks == []
    before = snapshots(output)
    assert load_run(output)[1] == cohort
    preview = run_extraction(cohort, models, output, kind="candidate", reference=reference,
                             gold_fingerprint="new", skipped_gold=skipped, resume=True, dry_run=True, provider_factory=factory)
    assert preview["models"]["ollama:qwen:4b"]["pending"] == int(not skip_new)
    assert preview["skipped_gold"] == skipped
    assert snapshots(output) == before
    if use_report:
        summary = report(output)
        assert summary["models"]["ollama:qwen:4b"]["pending"] == int(not skip_new)
        assert summary["skipped_gold"] == skipped
        assert not (output / APPEND_TRANSACTION).exists()
    assert run_extraction(cohort, models, output, kind="candidate", reference=reference,
                          gold_fingerprint="new", skipped_gold=skipped,
                          resume=True, provider_factory=factory, show_progress=False)["successful"]
    assert factory.calls == ([] if skip_new else [("ollama:qwen:4b", cohort[1].chunk_id)])
    assert read_jsonl(output / "predictions.jsonl")[0] == original[0]
    assert not (output / APPEND_TRANSACTION).exists()


def test_candidate_append_keeps_failed_predictions_in_micro_scores(cohort, tmp_path):
    output = tmp_path / "candidate"
    models = [settings("ollama")]
    reference = {r.chunk_id: validate_extraction_response(r, content(r)) for r in cohort}
    run_extraction(cohort[:1], models, output, kind="candidate", reference=reference,
                   gold_fingerprint="old", provider_factory=FakeFactory(lambda model, request: "invalid"), show_progress=False)
    original = read_jsonl(output / "predictions.jsonl")[0]
    factory = FakeFactory()
    summary = run_extraction(cohort, models, output, kind="candidate", reference=reference,
                             gold_fingerprint="new", append_start=2, provider_factory=factory, show_progress=False)
    local = summary["models"]["ollama:qwen:4b"]
    assert summary["complete"] and not summary["successful"] and local["failed"] == 1
    assert local["micro"]["entities"]["recall"] == 0.5
    assert local["micro"]["entities"]["precision"] == 1
    assert read_jsonl(output / "predictions.jsonl")[0] == original
    assert factory.calls == [("ollama:qwen:4b", cohort[1].chunk_id)]


def test_cli_compares_50_of_180_and_reports_130_gold_failures_offline(append_cli, tmp_path):
    source, gold, factory, invoke = append_cli
    cohort = load_paragraphs(source, count=180)
    failed_ids = {r.chunk_id for r in cohort[:130]}
    factory.behavior = lambda model, request: "malformed" if request.chunk_id in failed_ids else content(request)
    assert invoke("--paragraphs", "180")[0] == 1
    factory.behavior = None
    factory.calls.clear()
    factory.checks.clear()
    source.unlink()
    output = tmp_path / "candidate"
    parser = build_parser()
    arguments = ["run", "--gold", str(gold), "--output", str(output), "--no-progress",
                 "--model", "ollama:qwen:4b", "--model", "openrouter:candidate"]
    code, preview = execute(parser.parse_args([*arguments, "--dry-run"]))
    assert code == 0 and preview["requested"] == 180 and preview["comparable"] == 50
    assert preview["skipped"] == 130 and not output.exists()
    assert factory.calls == factory.checks == []

    code, summary = execute(parser.parse_args(arguments))
    assert code == 0 and summary["complete"] and summary["successful"]
    assert summary["requested"] == 180 and summary["comparable"] == 50 and summary["skipped"] == 130
    assert summary["skipped_gold"] == [
        {"chunk_id": r.chunk_id, "reason": "gold_failed", "error_type": "ValidationError"}
        for r in cohort[:130]]
    assert "message" not in json.dumps(summary["skipped_gold"])
    assert factory.calls == [(spec, r.chunk_id) for spec in ("ollama:qwen:4b", "openrouter:candidate")
                             for r in cohort[130:]]
    for model in summary["models"].values():
        assert model["expected"] == model["attempted"] == model["compared"] == 50
        assert model["failed"] == model["pending"] == 0
        assert model["micro"]["entities"]["f1"] == model["micro"]["relations"]["f1"] == 1
    assert len(read_jsonl(output / "cohort.jsonl")) == 180
    assert len(read_jsonl(output / "reference.jsonl")) == 50
    assert read_json(output / "manifest.json")["configuration"]["skipped_gold"] == summary["skipped_gold"]
    for path in gold.iterdir():
        path.unlink()
    gold.rmdir()
    assert {"output": str(output), **report(output)} == summary


@pytest.mark.parametrize("gold_status", ["failed", "missing"])
def test_all_unusable_gold_completes_without_provider_calls(cohort, tmp_path, monkeypatch, gold_status):
    import graphrag_pipeline.extraction_benchmark.command as command

    gold = tmp_path / "gold"
    if gold_status == "failed":
        gold_run(cohort, gold, FakeFactory(lambda model, request: "invalid"))
    else:
        def interrupt(model, request):
            raise KeyboardInterrupt
        with pytest.raises(KeyboardInterrupt):
            gold_run(cohort, gold, FakeFactory(interrupt))
    factory = FakeFactory()
    monkeypatch.setattr(command, "run_extraction", lambda *args, **kwargs: run_extraction(
        *args, **kwargs, provider_factory=factory))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    parser = build_parser()
    output = tmp_path / "candidate"
    arguments = ["run", "--gold", str(gold), "--output", str(output), "--no-progress",
                 "--model", "ollama:qwen:4b", "--model", "openrouter:candidate"]
    code, summary = execute(parser.parse_args(arguments))
    assert code == 0 and summary["complete"] and summary["successful"]
    assert summary["requested"] == summary["skipped"] == 2 and summary["comparable"] == 0
    assert [skip["reason"] for skip in summary["skipped_gold"]] == [f"gold_{gold_status}"] * 2
    for model in summary["models"].values():
        assert model["expected"] == model["attempted"] == model["compared"] == model["failed"] == model["pending"] == 0
        assert model["micro"]["entities"]["f1"] == 0
    assert factory.calls == factory.checks == []
    assert read_jsonl(output / "reference.jsonl") == read_jsonl(output / "predictions.jsonl") == []
    assert not (output / "preflight.json").exists()
    before = snapshots(output)
    assert execute(parser.parse_args([*arguments, "--resume"]))[1] == summary
    assert snapshots(output) == before
    assert factory.calls == factory.checks == []
    assert execute(parser.parse_args(["report", "--input", str(output)]))[1] == summary


@pytest.mark.parametrize("interrupted", [False, True])
def test_partial_gold_candidate_resume_and_retry_skip_only_gold_failures(cohort, tmp_path, monkeypatch, interrupted):
    import graphrag_pipeline.extraction_benchmark.command as command

    gold = tmp_path / "gold"
    gold_run(cohort, gold, FakeFactory(lambda model, request: "invalid" if request == cohort[0] else content(request)))
    def fail(model, request):
        if interrupted:
            raise KeyboardInterrupt
        return "invalid"
    factory = FakeFactory(fail)
    monkeypatch.setattr(command, "run_extraction", lambda *args, **kwargs: run_extraction(
        *args, **kwargs, provider_factory=factory))
    output = tmp_path / "candidate"
    parser = build_parser()
    arguments = ["run", "--gold", str(gold), "--output", str(output),
                 "--model", "ollama:qwen:4b", "--no-progress"]
    if interrupted:
        with pytest.raises(KeyboardInterrupt):
            execute(parser.parse_args(arguments))
    else:
        code, failed = execute(parser.parse_args(arguments))
        assert code == 1 and failed["models"]["ollama:qwen:4b"]["failed"] == 1
        assert failed["models"]["ollama:qwen:4b"]["micro"]["entities"]["fn"] == 2
    before = report(output)
    assert before["models"]["ollama:qwen:4b"]["pending"] == int(interrupted)
    factory.behavior = None
    factory.calls.clear()
    code, summary = execute(parser.parse_args([*arguments, "--resume", "--retry-failed"]))
    assert code == 0 and summary["skipped_gold"] == before["skipped_gold"]
    assert summary["models"]["ollama:qwen:4b"]["micro"]["entities"]["f1"] == 1
    assert factory.calls == [("ollama:qwen:4b", cohort[1].chunk_id)]

    # Repairing gold must not silently add a previously skipped paragraph.
    gold_run(cohort, gold, resume=True, retry_failed=True)
    before_files = snapshots(output)
    factory.calls.clear()
    factory.checks.clear()
    with pytest.raises(ValueError, match="resume configuration/source/gold mismatch"):
        execute(parser.parse_args([*arguments, "--resume"]))
    assert factory.calls == factory.checks == [] and snapshots(output) == before_files


@pytest.mark.parametrize("all_new_skipped", [False, True])
def test_partial_gold_append_uses_source_positions_and_freezes_skip_set(append_cli, tmp_path, all_new_skipped):
    source, gold, factory, invoke = append_cli
    cohort = load_paragraphs(source, count=180)
    failed_ids = {cohort[0].chunk_id, cohort[149].chunk_id}
    failed_ids.update(r.chunk_id for r in (cohort[150:] if all_new_skipped else cohort[150::2]))
    gold_behavior = lambda model, request: "invalid" if request.chunk_id in failed_ids else content(request)
    factory.behavior = gold_behavior
    invoke("--paragraphs", "150")
    factory.behavior = None
    parser = build_parser()
    output = tmp_path / "candidate"
    arguments = ["run", "--gold", str(gold), "--output", str(output),
                 "--model", "ollama:qwen:4b", "--no-progress"]
    assert execute(parser.parse_args(arguments))[0] == 0
    original = read_jsonl(output / "predictions.jsonl")
    frozen_skips = report(output)["skipped_gold"]
    factory.behavior = gold_behavior
    invoke("--additional-paragraphs", "30")
    factory.behavior = None
    factory.calls.clear()
    factory.checks.clear()
    code, summary = execute(parser.parse_args([*arguments, "--start-paragraph", "151", "--end-paragraph", "180"]))
    assert code == 0 and summary["requested"] == 180
    assert summary["comparable"] == (148 if all_new_skipped else 163)
    assert summary["skipped_gold"][:2] == frozen_skips
    assert read_jsonl(output / "predictions.jsonl")[:148] == original
    assert factory.calls == [("ollama:qwen:4b", r.chunk_id) for r in cohort[150:] if r.chunk_id not in failed_ids]
    if all_new_skipped:
        assert factory.checks == []
    assert {"output": str(output), **report(output)} == summary


def test_candidate_append_rejects_changes_to_previously_skipped_gold(cohort, tmp_path):
    gold = tmp_path / "gold"
    gold_run(cohort[:1], gold, FakeFactory(lambda model, request: "invalid"))
    requests, reference, fingerprint, skipped = load_gold(gold)
    output = tmp_path / "candidate"
    models = [settings("ollama")]
    run_extraction(requests, models, output, kind="candidate", reference=reference,
                   gold_fingerprint=fingerprint, skipped_gold=skipped, provider_factory=FakeFactory(), show_progress=False)
    gold_run(cohort, gold, append_start=2, resume=True, retry_failed=True)
    requests, reference, fingerprint, skipped = load_gold(gold)
    factory = FakeFactory()
    before = snapshots(output)
    with pytest.raises(ValueError, match="including skipped gold"):
        run_extraction(requests, models, output, kind="candidate", reference=reference,
                       gold_fingerprint=fingerprint, skipped_gold=skipped,
                       append_start=2, provider_factory=factory, show_progress=False)
    assert snapshots(output) == before and factory.calls == factory.checks == []


def test_all_skipped_candidate_can_append_usable_gold(cohort, tmp_path, monkeypatch):
    import graphrag_pipeline.extraction_benchmark.command as command

    gold = tmp_path / "gold"
    gold_run(cohort[:1], gold, FakeFactory(lambda model, request: "invalid"))
    factory = FakeFactory()
    monkeypatch.setattr(command, "run_extraction", lambda *args, **kwargs: run_extraction(
        *args, **kwargs, provider_factory=factory))
    parser = build_parser()
    output = tmp_path / "candidate"
    arguments = ["run", "--gold", str(gold), "--output", str(output),
                 "--model", "ollama:qwen:4b", "--no-progress"]
    assert execute(parser.parse_args(arguments))[0] == 0
    assert factory.calls == factory.checks == []
    gold_run(cohort, gold, append_start=2)
    code, summary = execute(parser.parse_args([*arguments, "--additional-paragraphs", "1"]))
    assert code == 0 and summary["requested"] == 2 and summary["skipped"] == summary["comparable"] == 1
    assert factory.calls == [("ollama:qwen:4b", cohort[1].chunk_id)]
    assert summary["models"]["ollama:qwen:4b"]["micro"]["entities"]["f1"] == 1
    assert {"output": str(output), **report(output)} == summary


@pytest.mark.parametrize("corruption", ["prediction", "duplicate_skip", "unknown_skip", "missing_skip", "error_type"])
def test_offline_report_rejects_inconsistent_partial_gold_artifacts(cohort, tmp_path, corruption):
    gold = tmp_path / "gold"
    gold_run(cohort, gold, FakeFactory(lambda model, request: "invalid" if request == cohort[0] else content(request)))
    requests, reference, fingerprint, skipped = load_gold(gold)
    output = tmp_path / "candidate"
    run_extraction(requests, [settings("ollama")], output, kind="candidate", reference=reference,
                   gold_fingerprint=fingerprint, skipped_gold=skipped, provider_factory=FakeFactory(), show_progress=False)
    if corruption == "prediction":
        rows = read_jsonl(output / "predictions.jsonl")
        rows[0]["chunk_id"] = cohort[0].chunk_id
        write_jsonl(output / "predictions.jsonl", rows)
    else:
        manifest = read_json(output / "manifest.json")
        skips = manifest["configuration"]["skipped_gold"]
        if corruption == "duplicate_skip": skips.append(dict(skips[0]))
        elif corruption == "unknown_skip": skips[0]["chunk_id"] = "unknown"
        elif corruption == "missing_skip": skips.clear()
        else: skips[0]["error_type"] = None
        (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        report(output)
