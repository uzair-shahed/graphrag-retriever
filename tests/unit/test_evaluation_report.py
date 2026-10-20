"""Artifact-only reporting, safe embedding and automatic finalization contracts."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphrag_pipeline.evaluation.command import build_parser, execute
from graphrag_pipeline.evaluation.records import DEFAULT_DATASET, load_cases
from graphrag_pipeline.evaluation.report import generate_report, load_report_data
from graphrag_pipeline.evaluation.runner import run_evaluation
from graphrag_pipeline.evaluation.runtime import EvaluationSettings
from graphrag_pipeline.query import EvidenceChunk


@pytest.fixture
def report_inputs(tmp_path):
    source = tmp_path / DEFAULT_DATASET.name
    source.write_text(Path("tests/fixtures/musique/valid_records.jsonl").read_text())
    case = load_cases(source, records=1)[0]
    chunks = [EvidenceChunk(chunk_id=f"musique:{case.gold.id}:paragraph:{p.idx}", text=p.paragraph_text)
              for p in case.gold.paragraphs]
    retriever = SimpleNamespace(embed=lambda q: [1., 0.], retrieve=lambda *a: chunks)
    llm = MagicMock()
    llm.invoke.return_value = SimpleNamespace(content=json.dumps({"answer": "Boston", "citations": [chunks[1].chunk_id]}))
    metrics = {name: SimpleNamespace(ascore=AsyncMock(return_value=SimpleNamespace(value=0., reason=None)))
               for name in ("faithfulness", "answer_relevancy", "context_relevance")}
    return case, retriever, llm, metrics


def evaluate(inputs, directory):
    case, retriever, llm, metrics = inputs
    return asyncio.run(run_evaluation([case], retriever, llm, metrics,
        EvaluationSettings(max_context_tokens=100000), directory, count_tokens=len,
        snapshot=lambda: {"sha256": "same"}, show_progress=False))


@pytest.fixture
def saved_run(report_inputs, tmp_path):
    directory = tmp_path / "evaluation"
    evaluate(report_inputs, directory)
    return directory


def edit_rows(directory, transform):
    path = directory / "records.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    transform(rows)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_automatic_report_preserves_scores_and_deduplicates_context(saved_run):
    data = load_report_data(saved_run)
    assert data["warnings"] == []
    assert data["summary"] == json.loads((saved_run / "summary.json").read_text())
    record = data["records"][0]
    assert len(record["results"]) == 4
    assert len(record["evidence"]) == 4  # Sixteen original copies pooled into four.
    original = json.loads((saved_run / "records.jsonl").read_text().splitlines()[0])
    row = record["results"]["record/vectorrag"]
    assert record["labels"] == original["labels"]
    assert [record["evidence"][i] for i in row["retrieval_evidence"]] == original["retrieval_evidence"]
    assert record["texts"][row["answer_context"]] == original["answer_context"]
    assert record["texts"][row["answer_prompt"]] == original["answer_prompt"]
    assert data["summary"]["groups"]["record/vectorrag"]["metrics"]["faithfulness"] == {"mean": 0., "count": 1}
    html = (saved_run / "report.html").read_text()
    assert "__REPORT_" not in html
    assert 'src="https://' not in html and 'href="https://' not in html


def test_offline_command_requires_no_dataset_graph_or_providers(saved_run, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("offline report must not load settings or source data")
    monkeypatch.setattr("graphrag_pipeline.evaluation.command.selection_from_args", forbidden)
    monkeypatch.setattr("graphrag_pipeline.evaluation.command.Neo4jSettings.from_env", forbidden)
    originals = {p.name: p.read_bytes() for p in saved_run.iterdir() if p.suffix != ".html"}
    args = build_parser().parse_args(["report", "--input", str(saved_run)])
    code, result = asyncio.run(execute(args))
    assert code == 0 and result["completed_rows"] == 4 and result["report"] == str(saved_run / "report.html")
    assert originals == {p.name: p.read_bytes() for p in saved_run.iterdir() if p.suffix != ".html"}


def test_abstentions_failures_and_unavailable_metrics_remain_visible(saved_run):
    def alter(rows):
        for row in rows[:2]:
            row["response"].update(status="insufficient_evidence", answer=None, citations=[])
            for metric in ("faithfulness", "answer_relevancy"):
                row["ragas"][metric] = {"value": None, "status": "not_applicable"}
        rows[1]["response"].update(status="failed", error="answer provider failed")
        rows[1]["graph_component_complete"] = False
        rows[1]["ragas"]["context_relevance"] = {"value": None, "status": "failed", "error": {"error": "judge failed"}}
    edit_rows(saved_run, alter)
    data = load_report_data(saved_run)
    vector, graph = (data["summary"]["groups"][f"record/{b}"] for b in ("vectorrag", "graphrag"))
    assert vector["metrics"]["faithfulness"] == {"mean": None, "count": 0}
    assert vector["abstention_rate"] == 1 and graph["failure_rate"] == 1
    assert graph["metric_failures"]["context_relevance"] == graph["traversal_incomplete"] == 1
    assert any("summary differs" in w for w in data["warnings"])


def test_partial_run_marks_missing_results_and_ignores_only_incomplete_tail(saved_run):
    edit_rows(saved_run, lambda rows: rows.__delitem__(slice(1, None)))
    manifest_path = saved_run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "interrupted"
    manifest["selected_records"].append({"id": "not-started", "source_line": 2})
    manifest_path.write_text(json.dumps(manifest))
    with (saved_run / "records.jsonl").open("a") as stream:
        stream.write('{"incomplete":')
    data = load_report_data(saved_run)
    assert data["summary"]["completed_rows"] == 1 and data["summary"]["expected_rows"] == 8
    assert data["records"][1]["labels"] is None and data["records"][1]["results"] == {}
    assert any("incomplete final" in w for w in data["warnings"])
    assert any("Partial run" in w for w in data["warnings"])
    assert generate_report(saved_run)["completed_rows"] == 1


@pytest.mark.parametrize("mutation", ["malformed", "malformed_no_newline", "schema", "duplicate", "question", "labels", "evidence", "context", "prompt", "nan", "scored_null"])
def test_rejects_corrupt_complete_rows_and_keeps_previous_report(saved_run, mutation):
    previous = (saved_run / "report.html").read_bytes()
    if mutation in ("malformed", "malformed_no_newline"):
        with (saved_run / "records.jsonl").open("a") as stream:
            stream.write('{"bad":\n' if mutation == "malformed" else '{"bad":}')
    else:
        def alter(rows):
            if mutation == "schema": del rows[0]["labels"]
            if mutation == "duplicate": rows.append(rows[0])
            if mutation == "question": rows[0]["response"]["question"] = "wrong question"
            if mutation == "labels": rows[1]["labels"]["answer"] = "inconsistent gold answer"
            if mutation == "evidence": rows[0]["answer_evidence"] = []
            if mutation == "context": rows[0]["answer_context"] = "wrong context"
            if mutation == "prompt": rows[0]["answer_prompt"] = "wrong prompt"
            if mutation == "nan": rows[0]["ragas"]["faithfulness"]["value"] = float("nan")
            if mutation == "scored_null": rows[0]["ragas"]["faithfulness"]["value"] = None
        edit_rows(saved_run, alter)
    with pytest.raises(ValueError):
        generate_report(saved_run)
    assert (saved_run / "report.html").read_bytes() == previous
    assert not list(saved_run.glob("*.tmp"))


def test_dataset_and_model_text_cannot_break_out_of_json_blocks(saved_run):
    attack = '</script><script>window.reportXSS=true</script><img src=x onerror="alert(1)">'
    def alter(rows):
        for row in rows:
            row["labels"]["question"] = row["request"]["question"] = row["response"]["question"] = attack
            row["response"]["answer"] = attack
    edit_rows(saved_run, alter)
    generate_report(saved_run)
    html = (saved_run / "report.html").read_text()
    assert attack not in html and "\\u003c/script\\u003e" in html
    assert load_report_data(saved_run)["records"][0]["labels"]["question"] == attack


def test_report_failure_preserves_completed_evaluation(report_inputs, tmp_path, monkeypatch):
    monkeypatch.setattr("graphrag_pipeline.evaluation.report.generate_report", MagicMock(side_effect=OSError("disk full")))
    output = tmp_path / "report-error"
    with pytest.raises(RuntimeError, match="HTML report generation failed"):
        evaluate(report_inputs, output)
    assert json.loads((output / "manifest.json").read_text())["status"] == "completed"
    assert json.loads((output / "summary.json").read_text())["completed_rows"] == 4


@pytest.mark.parametrize("report_fails", [False, True])
def test_interruption_generates_partial_report_without_masking_cause(report_inputs, tmp_path, monkeypatch, report_fails):
    case, retriever, llm, metrics = report_inputs
    original = retriever.retrieve
    def retrieve(backend, vector, record_id):
        if record_id is None:
            raise KeyboardInterrupt("original interruption")
        return original(backend, vector, record_id)
    retriever.retrieve = retrieve
    if report_fails:
        monkeypatch.setattr("graphrag_pipeline.evaluation.report.generate_report", MagicMock(side_effect=OSError("report failed")))
    output = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt, match="original interruption"):
        evaluate((case, retriever, llm, metrics), output)
    assert json.loads((output / "manifest.json").read_text())["status"] == "interrupted"
    assert len((output / "records.jsonl").read_text().splitlines()) == 2
    assert (output / "report.html").exists() is not report_fails


def test_missing_summary_reconstructed_and_valid_final_row_without_newline_accepted(saved_run):
    (saved_run / "summary.json").unlink()
    path = saved_run / "records.jsonl"
    path.write_text(path.read_text().rstrip("\n"))
    result = generate_report(saved_run)
    assert result["completed_rows"] == 4
    assert any("summary.json is absent" in w for w in result["warnings"])


def test_no_completed_rows_still_gets_a_partial_report(saved_run):
    (saved_run / "records.jsonl").write_text("")
    result = generate_report(saved_run)
    assert result["completed_rows"] == 0
    assert any("0 of 4" in warning for warning in result["warnings"])
    record = load_report_data(saved_run)["records"][0]
    assert record["labels"] is None and not record["results"]
