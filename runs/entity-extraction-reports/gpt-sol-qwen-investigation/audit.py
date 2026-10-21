"""Read-only diagnostic audit; separate metrics never replace the official scorer.

Run from the repository root with the project's Python environment.
"""
import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

from graphrag_pipeline.extraction_benchmark.artifacts import load_gold, load_run, read_json
from graphrag_pipeline.extraction_benchmark.scoring import score


def norm(value):
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join("".join(" " if unicodedata.category(c).startswith("P") else c for c in value).split())


def labelnorm(value):
    return norm(re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value))


def sets(result):
    if result is None:
        return {key: set() for key in ("entities_strict", "entities_surface", "entities_format", "relations_strict",
                                      "relations_untyped", "relations_format", "relations_pairs", "relations_undirected")}
    mentions = {m.mention_id: (norm(m.mention), norm(m.entity_type)) for m in result.mentions}
    endpoints = [(mentions[t.source_mention_id], t.relationship_type, mentions[t.target_mention_id]) for t in result.triples]
    return {
        "entities_strict": set(mentions.values()),
        "entities_surface": {m[0] for m in mentions.values()},
        "entities_format": {(norm(m.mention), labelnorm(m.entity_type)) for m in result.mentions},
        "relations_strict": {(s, norm(r), o) for s, r, o in endpoints},
        "relations_untyped": {(s[0], norm(r), o[0]) for s, r, o in endpoints},
        "relations_format": {(s[0], labelnorm(r), o[0]) for s, r, o in endpoints},
        "relations_pairs": {(s[0], o[0]) for s, _, o in endpoints},
        "relations_undirected": {tuple(sorted((s[0], o[0]))) for s, _, o in endpoints},
    }


def metric(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0}


def hashes(paths):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for directory in paths
            for p in sorted(directory.iterdir()) if p.is_file()}


def run(gold_path, candidate_path):
    before = hashes([gold_path, candidate_path])
    gold_manifest, gold_cohort, gold_rows, _ = load_run(gold_path)
    manifest, cohort, rows, frozen = load_run(candidate_path)
    requests, references, fingerprint, skips = load_gold(gold_path)
    assert cohort == gold_cohort == requests, "cohort mismatch"
    assert references == frozen and not skips, "gold reference mismatch or unusable labels"
    assert manifest["configuration"]["gold_fingerprint"] == fingerprint, "gold fingerprint mismatch"
    specs = [f"{m['provider']}:{m['model']}" for m in manifest["configuration"]["models"]]
    assert len(specs) == 1, "audit currently targets one selected model"
    by_id = {row.chunk_id: row for row in rows.values()}
    assert len(by_id) == len(cohort), "finish all attempts before this audit"
    counts = defaultdict(Counter)
    type_confusions, relation_disagreements = Counter(), Counter()
    labels = {side: {task: Counter() for task in ("types", "relations")} for side in ("gold", "qwen")}
    grounding = {side: {"total": 0, "raw_not_in_body": 0, "normalized_not_in_body": 0, "examples": []}
                 for side in ("gold", "qwen")}
    examples = []
    gold_edges_endpoints = Counter()
    failures = []
    conditional = Counter()
    for request in requests:
        row = by_id[request.chunk_id]
        gold, prediction = references[request.chunk_id], row.result
        gs, ps = sets(gold), sets(prediction)
        for key in gs:
            counts[key].update(tp=len(gs[key] & ps[key]), fp=len(ps[key] - gs[key]), fn=len(gs[key] - ps[key]))
        independently = {task: metric(len(gs[key] & ps[key]), len(ps[key] - gs[key]), len(gs[key] - ps[key]))
                         for task, key in (("entities", "entities_strict"), ("relations", "relations_strict"))}
        expected = score(gold, prediction)
        # IDs and list positions are local references, not evaluation labels.
        renamed = deepcopy(gold)
        names = {m.mention_id: f"audit-{index}" for index, m in enumerate(renamed.mentions)}
        for mention in renamed.mentions:
            mention.mention_id = names[mention.mention_id]
        for triple in renamed.triples:
            triple.source_mention_id = names[triple.source_mention_id]
            triple.target_mention_id = names[triple.target_mention_id]
        renamed.mentions.reverse()
        renamed.triples.reverse()
        assert sets(renamed) == gs
        identical = score(gold, renamed)
        assert identical['entities']['fp'] == identical['entities']['fn'] == 0
        assert identical['relations']['fp'] == identical['relations']['fn'] == 0
        for task in ("entities", "relations"):
            for key in ("tp", "fp", "fn", "precision", "recall", "f1"):
                assert abs(independently[task][key] - expected[task][key]) < 1e-12
                assert abs(independently[task][key] - row.scores[task][key]) < 1e-12
        gm, pm = defaultdict(set), defaultdict(set)
        for side, result, mapping in (("gold", gold, gm), ("qwen", prediction, pm)):
            if result is None:
                continue
            for mention in result.mentions:
                mapping[norm(mention.mention)].add(norm(mention.entity_type))
                labels[side]["types"][mention.entity_type] += 1
                grounding[side]["total"] += 1
                if mention.mention not in request.text:
                    grounding[side]["raw_not_in_body"] += 1
                if norm(mention.mention) not in norm(request.text):
                    grounding[side]["normalized_not_in_body"] += 1
                    grounding[side]["examples"].append({"chunk_id": request.chunk_id, "mention": mention.model_dump(),
                                                         "text": request.text, "title": request.title})
            labels[side]["relations"].update(t.relationship_type for t in result.triples)
        confusions = []
        for text in gm.keys() & pm.keys():
            if not gm[text] & pm[text]:
                signature = (" / ".join(sorted(gm[text])), " / ".join(sorted(pm[text])))
                type_confusions[signature] += 1
                confusions.append({"mention": text, "gold": sorted(gm[text]), "qwen": sorted(pm[text])})
        grels, prels = defaultdict(set), defaultdict(set)
        for s, label, o in gs["relations_untyped"]:
            grels[s, o].add(label)
        for s, label, o in ps["relations_untyped"]:
            prels[s, o].add(label)
        for pair in grels.keys() & prels.keys():
            g, p = grels[pair], prels[pair]
            conditional.update(tp=len(g & p), fp=len(p - g), fn=len(g - p))
            if not g & p:
                relation_disagreements[(" / ".join(sorted(g)), " / ".join(sorted(p)))] += 1
        for source, target in gs["relations_pairs"]:
            gold_edges_endpoints["total"] += 1
            gold_edges_endpoints["both_surfaces_available"] += source in pm and target in pm
            gold_edges_endpoints["both_typed_available"] += bool(gm[source] & pm[source]) and bool(gm[target] & pm[target])
        reverse_only = [(s, o) for s, o in ps["relations_pairs"] if (o, s) in gs["relations_pairs"] and (s, o) not in gs["relations_pairs"]]
        gold_edges_endpoints["qwen_reverse_only_pairs"] += len(reverse_only)
        if row.status == "failed":
            failures.append({"chunk_id": request.chunk_id, "error": row.error, "metadata": row.metadata,
                             "raw_content": row.raw_content, "gold_entities": len(gs["entities_strict"]),
                             "gold_relations": len(gs["relations_strict"])})
        gmentions = {m.mention_id: m for m in gold.mentions}
        pmentions = {m.mention_id: m for m in prediction.mentions} if prediction else {}
        def triples(result, mentions):
            return [] if result is None else [
                [mentions[t.source_mention_id].mention, t.relationship_type, mentions[t.target_mention_id].mention]
                for t in result.triples]
        examples.append({"index": len(examples) + 1, "chunk_id": request.chunk_id, "record_id": request.record_id,
                         "title": request.title, "text": request.text, "status": row.status,
                         "format": row.response_format_version, "official": independently,
                         "gold_entities": [m.model_dump() for m in gold.mentions],
                         "qwen_entities": [m.model_dump() for m in prediction.mentions] if prediction else [],
                         "gold_relations": triples(gold, gmentions), "qwen_relations": triples(prediction, pmentions),
                         "type_disagreements": confusions,
                         "same_endpoint_label_disagreements": [
                             {"source": s, "target": o, "gold": sorted(grels[s, o]), "qwen": sorted(prels[s, o])}
                             for s, o in grels.keys() & prels.keys() if not grels[s, o] & prels[s, o]]})
    metrics = {key: metric(c["tp"], c["fp"], c["fn"]) for key, c in counts.items()}
    upper_bound = {}
    for task in ("entities", "relations"):
        baseline = metrics[f"{task}_strict"]
        recovered = sum(row[f"gold_{task}"] for row in failures)
        upper_bound[task] = metric(baseline["tp"] + recovered, baseline["fp"], baseline["fn"] - recovered)
    formats = Counter(str(row.response_format_version) for row in rows.values())
    migrations = {}
    for side, path in (("gold", gold_path), ("qwen", candidate_path)):
        audit = read_json(path / "migration.json")
        original = audit["source_manifest"]["configuration"]
        migrations[side] = {"source_run_id": audit["source_manifest"]["run_id"],
                            "source_prompt": original["prompt"], "source_models": original["models"],
                            "preflight": read_json(path / "preflight.json")}
    group_metrics = {}
    for version in sorted(formats):
        selected = [e for e in examples if str(e["format"]) == version]
        group_metrics[version] = {"paragraphs": len(selected), **{
            task: metric(*(sum(e["official"][task][key] for e in selected) for key in ("tp", "fp", "fn")))
            for task in ("entities", "relations")}}
    record_ids = sorted({r.record_id for r in requests})
    texts = Counter(r.text for r in requests)
    after = hashes([gold_path, candidate_path])
    assert before == after, "inputs changed during audit"
    report = {"source_hashes": before, "inputs_unchanged": True, "independent_scorer_agrees_all_rows": True,
              "id_and_order_invariance_all_rows": True,
              "code_hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                  Path('src/graphrag_pipeline/extraction_benchmark/scoring.py'),
                  Path('src/graphrag_pipeline/extraction_benchmark/providers.py'))},
              "gold_manifest": gold_manifest, "candidate_manifest": manifest,
              "cohort": {"paragraphs": len(requests), "records": len(record_ids), "unique_texts": len(texts),
                         "duplicate_occurrences": sum(c - 1 for c in texts.values())},
              "metrics": metrics, "conditional_label_metrics": metric(conditional["tp"], conditional["fp"], conditional["fn"]),
              "endpoint_availability": dict(gold_edges_endpoints), "failure_perfect_replacement_bound": upper_bound,
              "statuses": dict(Counter(r.status for r in rows.values())), "formats": dict(formats), "by_format": group_metrics,
              "labels": {side: {task: counter.most_common() for task, counter in tasks.items()} for side, tasks in labels.items()},
              "type_confusions": [[*pair, count] for pair, count in type_confusions.most_common()],
              "relation_disagreements": [[*pair, count] for pair, count in relation_disagreements.most_common()],
              "grounding": grounding, "failures": failures, "migrations": migrations, "examples": examples}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=Path("runs/entity-extraction/gold-run-180-v2"))
    parser.add_argument("--candidate", type=Path, default=Path("runs/entity-extraction/qwen3-4b-instruct-v2"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("evidence.json"))
    args = parser.parse_args()
    for source in (args.gold.resolve(), args.candidate.resolve()):
        if source == args.output.resolve() or source in args.output.resolve().parents:
            parser.error('--output must be separate from benchmark inputs')
    result = run(args.gold.resolve(), args.candidate.resolve())
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("cohort", "metrics", "statuses", "formats", "endpoint_availability",
                                                 "conditional_label_metrics", "failure_perfect_replacement_bound")}, indent=2))
