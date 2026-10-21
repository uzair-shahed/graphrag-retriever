"""Comparison metrics, reasoning pairs, artifact isolation and portable report output."""

import base64
import json
from copy import deepcopy
from dataclasses import replace
from html.parser import HTMLParser
from xml.etree import ElementTree

import pytest

from graphrag_pipeline.extraction_benchmark.artifacts import (
    load_gold, read_json, read_jsonl, write_json, write_jsonl,
)
from graphrag_pipeline.extraction_benchmark.command import build_parser, execute
from graphrag_pipeline.extraction_benchmark.contracts import validate_extraction_response
from graphrag_pipeline.extraction_benchmark.providers import Generation, ModelSettings
from graphrag_pipeline.extraction_benchmark.records import load_paragraphs
from graphrag_pipeline.extraction_benchmark.runner import run_extraction
from graphrag_pipeline.extraction_benchmark.visualization import FILES, visualize


def payload():
    return {"mentions": [{"mention_id": "a", "mention": "Alice", "entity_type": "Person"},
                         {"mention_id": "b", "mention": "Bob", "entity_type": "Person"}],
            "triples": [{"source_mention_id": "a", "target_mention_id": "b", "relationship_type": "MET"}]}


class Factory:
    def __init__(self, failure=None):
        self.failure = failure

    def __call__(self, settings):
        failure = self.failure

        class Client:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def check(self): return {"mock": True}
            def generate(self, request):
                if failure is not None and request.paragraph_idx == failure:
                    raise ValueError("saved extraction failure")
                return Generation(json.dumps(payload()), {"finish_reason": "stop"})

        return Client()


@pytest.fixture
def experiment(tmp_path):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(json.dumps({"id": "record", "paragraphs": [
        {"idx": idx, "title": "Meeting", "paragraph_text": "Alice met Bob."} for idx in range(3)
    ]}), encoding="utf-8")
    requests = load_paragraphs(dataset, count=3)
    gold = tmp_path / "gold"
    gold_settings = ModelSettings("openrouter", "gold", "http://provider.test", reasoning_effort="high")
    run_extraction(requests, [gold_settings], gold, kind="gold", provider_factory=Factory(failure=2), show_progress=False)
    cohort, reference, fingerprint, skipped = load_gold(gold)
    settings = ModelSettings("ollama", "qwen:4b", "http://provider.test", think=False)

    def candidate(name, models=None, failure=None, selected=None, labels=None):
        path = tmp_path / name
        run_extraction(selected or cohort, models or [settings], path, kind="candidate",
                       reference=reference if labels is None else labels, gold_fingerprint=fingerprint,
                       skipped_gold=skipped if selected is None else [],
                       provider_factory=Factory(failure=failure), show_progress=False)
        return path

    return gold, candidate, settings


def snapshot(path):
    return {p.name: p.read_bytes() for p in path.iterdir()}


def test_cli_metrics_skips_failures_and_input_preservation(experiment, tmp_path, monkeypatch):
    gold, candidate, _ = experiment
    off = candidate("off", failure=1)
    # Summaries must not be the source of truth; neither file is required to build the report.
    (gold / "summary.json").unlink()
    (off / "summary.json").write_text("stale summary", encoding="utf-8")
    before = [snapshot(gold), snapshot(off)]
    monkeypatch.setattr("graphrag_pipeline.extraction_benchmark.providers.JSONProvider.__init__",
                        lambda *args, **kwargs: pytest.fail("visualization called a provider"))
    output = tmp_path / "report"
    code, result = execute(build_parser().parse_args([
        "visualize", "--gold", str(gold), "--run", f"Qwen-off={off}", "--output", str(output)]))
    assert code == 0 and result["provider_calls"] == 0 and result["models_and_runs"] == 1
    assert [snapshot(gold), snapshot(off)] == before
    data = read_json(output / "comparison.json")
    assert data["gold"]["requested"] == 3 and data["gold"]["comparable"] == 2 and data["gold"]["skipped"] == 1
    series = data["series"][0]
    assert (series["succeeded"], series["failed"], series["pending"]) == (1, 1, 0)
    assert series["micro"]["entities"] == {"tp": 2, "fp": 0, "fn": 2, "precision": 1.0, "recall": 0.5, "f1": 2 / 3}
    assert series["micro"]["relations"] == {"tp": 1, "fp": 0, "fn": 1, "precision": 1.0, "recall": 0.5, "f1": 2 / 3}
    assert "66.67%" in (output / "entities.svg").read_text(encoding="utf-8")
    assert "No completed on/off pair" in (output / "reasoning-delta.svg").read_text(encoding="utf-8")
    assert set(snapshot(output)) == set(FILES)


def test_multiple_models_and_repeated_runs_keep_all_reasoning_pairs(experiment, tmp_path):
    gold, candidate, settings = experiment
    off = candidate("off", failure=1)
    on = candidate("on", [replace(settings, think=True), replace(settings, model="another", think=None)])
    another_on = candidate("high", [replace(settings, think="high", max_tokens=2048)])
    hosted = candidate("hosted", [ModelSettings("openrouter", "hosted", "http://provider.test", reasoning_effort="none"),
                                   ModelSettings("openrouter", "default", "http://provider.test")])
    hosted_on = candidate("hosted-on", [ModelSettings("openrouter", "hosted", "http://provider.test", reasoning_effort="medium")])
    visualize(gold, [f"Off={off}", f"On={on}", f"High={another_on}", f"Hosted={hosted}", f"Hosted-on={hosted_on}"], tmp_path / "report")
    data = read_json(tmp_path / "report/comparison.json")
    assert len(data["series"]) == 7
    assert [item["reasoning"] for item in data["series"]] == ["off", "on", "unspecified", "on", "off", "unspecified", "on"]
    assert data["series"][3]["reasoning_label"] == "on (high)"
    assert data["series"][-1]["reasoning_label"] == "on (medium)"
    assert len(data["reasoning_deltas"]) == 3
    for pair in data["reasoning_deltas"][:2]:
        assert pair["entities"] == pytest.approx(100 / 3)
        assert pair["relations"] == pytest.approx(100 / 3)
    assert data["reasoning_deltas"][0]["changed_settings"] == []
    assert data["reasoning_deltas"][1]["changed_settings"] == ["max_tokens"]
    assert data["reasoning_deltas"][2]["entities"] == data["reasoning_deltas"][2]["relations"] == 0


def test_pending_rows_are_visible_and_not_paired(experiment, tmp_path):
    gold, candidate, settings = experiment
    off, on = candidate("off"), candidate("on", [replace(settings, think=True)])
    rows = read_jsonl(on / "predictions.jsonl")
    write_jsonl(on / "predictions.jsonl", rows[:1])
    visualize(gold, [f"Off={off}", f"On={on}"], tmp_path / "report")
    data = read_json(tmp_path / "report/comparison.json")
    assert data["series"][1]["pending"] == 1
    assert data["reasoning_deltas"] == []
    assert any("pending paragraphs" in text for text in data["warnings"])


def test_negative_reasoning_delta_is_signed(experiment, tmp_path):
    gold, candidate, settings = experiment
    off = candidate("off")
    on = candidate("on", [replace(settings, think=True)], failure=1)
    visualize(gold, [f"Off={off}", f"On={on}"], tmp_path / "report")
    data = read_json(tmp_path / "report/comparison.json")
    assert data["reasoning_deltas"][0]["entities"] == pytest.approx(-100 / 3)
    assert "-33.33 pp" in (tmp_path / "report/reasoning-delta.svg").read_text(encoding="utf-8")


@pytest.mark.parametrize("marker", [".benchmark.lock", ".append-transaction.json"])
def test_active_or_unfinished_runs_are_rejected_without_writes(experiment, tmp_path, marker):
    gold, candidate, _ = experiment
    run = candidate("candidate")
    (run / marker).write_text("unfinished", encoding="utf-8")
    with pytest.raises(ValueError, match="finish/resume"):
        visualize(gold, [f"Qwen={run}"], tmp_path / "report")
    assert not (tmp_path / "report").exists()


def test_legacy_gold_requires_migration(experiment, tmp_path):
    gold, candidate, _ = experiment
    run = candidate("candidate")
    manifest = read_json(gold / "manifest.json")
    manifest["format_version"] = 1
    write_json(gold / "manifest.json", manifest)
    with pytest.raises(ValueError, match="migrate --input"):
        visualize(gold, [f"Qwen={run}"], tmp_path / "report")


@pytest.mark.parametrize("corruption,match", [
    ("fingerprint", "gold fingerprint"), ("v1", "migrate --input"), ("scores", "saved scores"),
    ("cohort", "cohort differs"), ("reference", "frozen reference labels"), ("kind", "candidate run"),
])
def test_rejects_invalid_or_incompatible_artifacts(experiment, tmp_path, corruption, match):
    gold, candidate, _ = experiment
    run = candidate("candidate")
    if corruption in {"fingerprint", "v1"}:
        manifest = read_json(run / "manifest.json")
        if corruption == "fingerprint":
            manifest["configuration"]["gold_fingerprint"] = "other-gold"
        else:
            manifest["format_version"] = 1
        write_json(run / "manifest.json", manifest)
    elif corruption == "scores":
        rows = read_jsonl(run / "predictions.jsonl")
        rows[0]["scores"]["entities"]["tp"] = 999
        write_jsonl(run / "predictions.jsonl", rows)
    elif corruption == "cohort":
        requests, _, _, _ = load_gold(gold)
        run = candidate("prefix", selected=requests[:1])
    elif corruption == "reference":
        requests, reference, _, _ = load_gold(gold)
        data = deepcopy(payload())
        data["triples"] = []
        reference[requests[0].chunk_id] = validate_extraction_response(requests[0], json.dumps(data))
        run = candidate("wrong-reference", labels=reference)
    else:
        run = gold
    output = tmp_path / "report"
    with pytest.raises(ValueError, match=match):
        visualize(gold, [f"Candidate={run}"], output)
    assert not output.exists()


class ReportParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.downloads = []
        self.external = []
        self.scripts = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "script":
            self.scripts.append(attrs)
        if "download" in attrs:
            self.downloads.append(attrs["href"])
        if tag in {"script", "link", "img"}:
            self.external.append(attrs)


def test_portable_html_svg_escaping_and_owned_output_replacement(experiment, tmp_path):
    gold, candidate, _ = experiment
    run = candidate("candidate=path")
    output = tmp_path / "report"
    output.mkdir()
    note = output / "notes.txt"
    note.write_text("my notes", encoding="utf-8")
    label = '<script>alert("x")</script> & Qwen'
    visualize(gold, [f"{label}={run}"], output)
    html = (output / "index.html").read_text(encoding="utf-8")
    assert label not in html and '&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; &amp; Qwen' in html
    parser = ReportParser()
    parser.feed(html)
    assert parser.external == parser.scripts == []
    assert len(parser.downloads) == 4
    for url in parser.downloads:
        assert url.startswith("data:image/svg+xml;base64,")
        ElementTree.fromstring(base64.b64decode(url.partition(",")[2]))
    for name in FILES:
        if name.endswith(".svg"):
            root = ElementTree.parse(output / name).getroot()
            assert root.tag == "{http://www.w3.org/2000/svg}svg"
    first = snapshot(output)
    visualize(gold, [f"{label}={run}"], output)
    assert snapshot(output) == first
    assert note.read_text(encoding="utf-8") == "my notes"


def test_output_collisions_and_input_overlap_are_rejected(experiment, tmp_path):
    gold, candidate, _ = experiment
    run = candidate("candidate")
    for output in (gold, run, gold / "charts", gold.parent):
        with pytest.raises(ValueError, match="without nesting"):
            visualize(gold, [f"Qwen={run}"], output)
    output = tmp_path / "report"
    output.mkdir()
    original = output / "index.html"
    original.write_text("user-owned page", encoding="utf-8")
    with pytest.raises(ValueError, match="not owned"):
        visualize(gold, [f"Qwen={run}"], output)
    assert snapshot(output) == {"index.html": b"user-owned page"}


@pytest.mark.parametrize("values", [[], ["bad"], ["=path"], ["name="], ["a=path", "a=other"], ["a=path", "b=path"]])
def test_input_argument_errors(tmp_path, values):
    with pytest.raises(ValueError):
        visualize(tmp_path / "gold", values, tmp_path / "report")
