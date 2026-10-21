"""Offline preservation, failure recovery, format isolation and resumed generation."""

import json
from copy import deepcopy

import pytest

from graphrag_pipeline.extraction_benchmark.artifacts import (
    APPEND_TRANSACTION, digest, load_gold, load_run, now, read_json, read_jsonl,
    write_json, write_jsonl,
)
from graphrag_pipeline.extraction_benchmark.command import build_parser, execute
from graphrag_pipeline.extraction_benchmark.contracts import validate_extraction_response
from graphrag_pipeline.extraction_benchmark.migration import migrate_run
from graphrag_pipeline.extraction_benchmark.providers import ModelSettings
from graphrag_pipeline.extraction_benchmark.records import load_paragraphs
from graphrag_pipeline.extraction_benchmark.runner import report, run_extraction
from graphrag_pipeline.extraction_benchmark.scoring import SCORING_VERSION, score


def payload():
    return {"mentions": [{"mention_id": "alice", "mention": "Alice", "entity_type": "Person"},
                         {"mention_id": "bob", "mention": "Bob", "entity_type": "Person"}],
            "triples": [{"source_mention_id": "alice", "relationship_type": "MET", "target_mention_id": "bob"}]}


def legacy_payload():
    data = payload()
    for item in data["mentions"] + data["triples"]:
        item["evidence_span"] = {"start": 99, "end": 1, "text": "incorrect evidence"}
    data["triples"][0]["confidence"] = 0.99
    return data


def settings(kind):
    return ModelSettings("openrouter" if kind == "gold" else "ollama", "model", "http://provider.test",
                         reasoning_effort="high" if kind == "gold" else None)


def snapshot(path):
    return {p.name: p.read_bytes() for p in path.iterdir()}


@pytest.fixture
def legacy(tmp_path):
    dataset = tmp_path / "source.jsonl"
    dataset.write_text(json.dumps({"id": "record", "paragraphs": [
        {"idx": i, "title": "Meeting", "paragraph_text": "Alice met Bob."} for i in range(4)]}), encoding="utf-8")
    cohort = load_paragraphs(dataset, count=4)
    gold, candidate = tmp_path / "old-gold", tmp_path / "old-candidate"
    for path, kind in ((gold, "gold"), (candidate, "candidate")):
        path.mkdir()
        requests = cohort[:3]
        references = {}
        for request in requests[:2]:
            references[request.chunk_id] = validate_extraction_response(request, json.dumps(payload()))
        raw_reference = [{**r.model_dump(mode="json"), **legacy_payload()} for r in references.values()]
        config = {"paragraphs": 3, "cohort_sha256": digest([r.model_dump(mode="json") for r in requests]),
                  "models": [settings(kind).public()], "prompt": "original prompt with evidence",
                  "response_schema": {"type": "object", "description": "original v1 schema"},
                  "scoring_version": SCORING_VERSION}
        if kind == "candidate":
            config.update(reference_sha256=digest(raw_reference), gold_fingerprint="old-fingerprint",
                          skipped_gold=[{"chunk_id": requests[2].chunk_id, "reason": "gold_failed", "error_type": "ValueError"}])
            write_jsonl(path / "reference.jsonl", raw_reference)
        manifest = {"format_version": 1, "kind": kind, "run_id": kind, "created_at": now(), "configuration": config}
        rows = []
        for i, request in enumerate(requests if kind == "gold" else requests[:2]):
            succeeded = i < (2 if kind == "gold" else 1)
            result = {**validate_extraction_response(request, json.dumps(payload())).model_dump(mode="json"),
                      **legacy_payload()} if succeeded else None
            rows.append({"model": settings(kind).spec, "chunk_id": request.chunk_id,
                         "status": "succeeded" if succeeded else "failed", "result": result,
                         "raw_content": json.dumps(legacy_payload()),
                         "error": None if succeeded else {"type": "ValueError", "message": "bad evidence offsets"},
                         "attempts": 2, "timestamp": now(), "elapsed_seconds": 1.5,
                         "metadata": {"finish_reason": "stop", "usage": {"cost": 0.123}},
                         "scores": score(references[request.chunk_id], references[request.chunk_id] if succeeded else None)
                                   if kind == "candidate" else None})
        write_json(path / "manifest.json", manifest)
        write_jsonl(path / "cohort.jsonl", [r.model_dump(mode="json") for r in requests])
        write_jsonl(path / ("gold.jsonl" if kind == "gold" else "predictions.jsonl"), rows)
    return cohort, gold, candidate


class Factory:
    def __init__(self):
        self.calls = []

    def __call__(self, setting):
        from graphrag_pipeline.extraction_benchmark.providers import Generation
        owner = self
        class Client:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def check(self): return {"mock": True}
            def generate(self, request):
                owner.calls.append(request.chunk_id)
                return Generation(json.dumps(payload()), {"finish_reason": "stop"})
        return Client()


def test_gold_conversion_and_preview_preserve_paid_work(legacy, tmp_path, monkeypatch):
    cohort, gold, _ = legacy
    before = snapshot(gold)
    output = tmp_path / "new-gold"
    def forbidden(*args, **kwargs):
        raise AssertionError("offline conversion contacted a provider")
    monkeypatch.setattr("graphrag_pipeline.extraction_benchmark.providers.JSONProvider.__init__", forbidden)
    preview = migrate_run(gold, output, dry_run=True)
    assert preview["successful"] and preview["conversion"]["recovered_rows"] == 1
    assert not output.exists() and snapshot(gold) == before
    code, converted = execute(build_parser().parse_args([
        "migrate", "--input", str(gold), "--output", str(output)]))
    assert code == 0 and converted["successful"]
    assert converted["conversion"]["provider_calls"] == 0
    requests, reference, _, skipped = load_gold(output)
    assert requests == cohort[:3] and len(reference) == 3 and skipped == []
    _, _, rows, _ = load_run(output)
    for row, old in zip(rows.values(), read_jsonl(gold / "gold.jsonl"), strict=True):
        assert row.raw_content == old["raw_content"]
        assert row.metadata == old["metadata"]
        assert row.attempts == old["attempts"]
        assert row.timestamp == old["timestamp"]
        assert row.elapsed_seconds == old["elapsed_seconds"]
        assert row.response_format_version == 1
    audit = read_json(output / "migration.json")
    assert audit["source_manifest"]["configuration"]["prompt"] == "original prompt with evidence"
    assert audit["rows"][-1]["error"]["message"] == "bad evidence offsets"
    assert report(output)["successful"]
    assert snapshot(gold) == before


def test_candidate_recovery_full_coverage_resume_and_append(legacy, tmp_path):
    cohort, gold, candidate = legacy
    gold_v2, candidate_v2 = tmp_path / "gold-v2", tmp_path / "candidate-v2"
    migrate_run(gold, gold_v2)
    before = snapshot(candidate)
    converted = migrate_run(candidate, candidate_v2, gold=gold_v2)
    model = converted["models"][settings("candidate").spec]
    assert converted["requested"] == converted["comparable"] == 3
    assert converted["skipped"] == 0
    assert (model["attempted"], model["succeeded"], model["pending"]) == (2, 2, 1)
    requests, reference, fingerprint, skips = load_gold(gold_v2)
    factory = Factory()
    summary = run_extraction(requests, [settings("candidate")], candidate_v2, kind="candidate",
                             reference=reference, gold_fingerprint=fingerprint, skipped_gold=skips,
                             resume=True, show_progress=False, provider_factory=factory)
    assert summary["successful"]
    assert factory.calls == [cohort[2].chunk_id]
    assert summary["models"][settings("candidate").spec]["generation_formats"] == {"legacy_v1": 2, "simplified_v2": 1}
    assert report(candidate_v2) == summary
    run_extraction(cohort, [settings("gold")], gold_v2, kind="gold", append_start=4,
                   show_progress=False, provider_factory=Factory())
    requests, reference, fingerprint, skips = load_gold(gold_v2)
    summary = run_extraction(requests, [settings("candidate")], candidate_v2, kind="candidate",
                             reference=reference, gold_fingerprint=fingerprint, skipped_gold=skips,
                             append_start=4, show_progress=False, provider_factory=Factory())
    assert summary["successful"] and report(candidate_v2) == summary
    assert snapshot(candidate) == before


@pytest.mark.parametrize("failure", ["unknown_reference", "malformed", "transport", "incomplete"])
def test_nonretired_failures_remain_failed_and_retry_is_explicit(legacy, tmp_path, failure):
    cohort, gold, candidate = legacy
    path = candidate / "predictions.jsonl"
    rows = read_jsonl(path)
    if failure == "unknown_reference":
        data = legacy_payload()
        data["triples"][0]["target_mention_id"] = "absent"
        rows[1]["raw_content"] = json.dumps(data)
    elif failure == "malformed":
        rows[1]["raw_content"] = "not json"
    elif failure == "transport":
        rows[1]["raw_content"] = None
        rows[1]["error"] = {"type": "ValueError", "message": "transport failure"}
    else:
        rows[1]["error"] = {"type": "IncompleteGeneration", "message": "length"}
        rows[1]["metadata"]["finish_reason"] = "length"
    write_jsonl(path, rows)
    gold_v2, candidate_v2 = tmp_path / "gold-v2", tmp_path / "candidate-v2"
    migrate_run(gold, gold_v2)
    summary = migrate_run(candidate, candidate_v2, gold=gold_v2)
    assert summary["models"][settings("candidate").spec]["failed"] == 1
    assert report(candidate_v2)["models"][settings("candidate").spec]["failed"] == 1
    requests, reference, fingerprint, _ = load_gold(gold_v2)
    factory = Factory()
    run_extraction(requests, [settings("candidate")], candidate_v2, kind="candidate", reference=reference,
                   gold_fingerprint=fingerprint, resume=True, provider_factory=factory, show_progress=False)
    assert factory.calls == [cohort[2].chunk_id]
    factory.calls.clear()
    summary = run_extraction(requests, [settings("candidate")], candidate_v2, kind="candidate", reference=reference,
                            gold_fingerprint=fingerprint, resume=True, retry_failed=True,
                            provider_factory=factory, show_progress=False)
    assert summary["successful"] and factory.calls == [cohort[1].chunk_id]
    assert report(candidate_v2) == summary


@pytest.mark.parametrize("corruption", ["cohort", "reference", "scores", "result", "duplicate"])
def test_migration_rejects_corrupt_source_without_publishing(legacy, tmp_path, corruption):
    _, gold, candidate = legacy
    gold_v2 = tmp_path / "gold-v2"
    migrate_run(gold, gold_v2)
    if corruption == "cohort":
        rows = read_jsonl(candidate / "cohort.jsonl")
        rows[0]["text"] = "changed"
        write_jsonl(candidate / "cohort.jsonl", rows)
    elif corruption == "reference":
        rows = read_jsonl(candidate / "reference.jsonl")
        rows[0]["mentions"][0]["mention"] = "changed"
        write_jsonl(candidate / "reference.jsonl", rows)
    else:
        rows = read_jsonl(candidate / "predictions.jsonl")
        if corruption == "scores": rows[0]["scores"]["entities"]["tp"] += 1
        elif corruption == "result": rows[0]["result"]["mentions"][0]["mention"] = "changed"
        else: rows.append(deepcopy(rows[0]))
        write_jsonl(candidate / "predictions.jsonl", rows)
    before = snapshot(candidate)
    output = tmp_path / "new-candidate"
    with pytest.raises(ValueError): migrate_run(candidate, output, gold=gold_v2)
    assert not output.exists() and snapshot(candidate) == before


def test_locks_legacy_guidance_and_failed_staging_are_safe(legacy, tmp_path, monkeypatch):
    _, gold, candidate = legacy
    with pytest.raises(ValueError, match="migrate --input"):
        load_run(gold)
    for dry in (False, True):
        (gold / ".benchmark.lock").write_text("active")
        with pytest.raises(ValueError, match="locked"):
            migrate_run(gold, tmp_path / "new", dry_run=dry)
        (gold / ".benchmark.lock").unlink()
    with pytest.raises(ValueError, match="requires --gold"):
        migrate_run(candidate, tmp_path / "new")
    with pytest.raises(ValueError, match="without nesting"):
        migrate_run(gold, gold / "nested")
    (gold / APPEND_TRANSACTION).write_text("{}")
    with pytest.raises(ValueError, match="append transaction"):
        migrate_run(gold, tmp_path / "new")
    (gold / APPEND_TRANSACTION).unlink()
    before = snapshot(gold)
    monkeypatch.setattr("graphrag_pipeline.extraction_benchmark.migration.load_run",
                        lambda *args: (_ for _ in ()).throw(ValueError("staging rejected")))
    with pytest.raises(ValueError, match="staging rejected"):
        migrate_run(gold, tmp_path / "new")
    assert not (tmp_path / "new").exists() and snapshot(gold) == before
    assert not list(tmp_path.glob(".new.*"))


@pytest.mark.parametrize("corruption", ["raw", "audit", "unmarked_legacy"])
def test_migrated_report_checks_archived_response_and_audit(legacy, tmp_path, corruption):
    _, gold, _ = legacy
    output = tmp_path / "new"
    migrate_run(gold, output)
    if corruption == "audit":
        audit = read_json(output / "migration.json")
        audit["source_manifest"]["run_id"] = "changed"
        write_json(output / "migration.json", audit)
    else:
        rows = read_jsonl(output / "gold.jsonl")
        if corruption == "raw": rows[0]["raw_content"] += " "
        else: rows[0]["response_format_version"] = 2
        write_jsonl(output / "gold.jsonl", rows)
    with pytest.raises(ValueError): report(output)
