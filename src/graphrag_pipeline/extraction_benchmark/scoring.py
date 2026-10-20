"""Deterministic set comparison; no embedding, graph, or LLM judge."""

from __future__ import annotations

import unicodedata

from graphrag_pipeline.extraction.schemas import ExtractionResult

SCORING_VERSION = "normalized-exact-v1"


def normalize(value: str) -> str:
    """NFKC/casefold, replace Unicode punctuation with spaces, collapse spaces."""
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join("".join(" " if unicodedata.category(c).startswith("P") else c for c in value).split())


def extraction_sets(result: ExtractionResult | None) -> tuple[set, set]:
    if result is None:
        return set(), set()
    mentions = {m.mention_id: (normalize(m.mention), normalize(m.entity_type)) for m in result.mentions}
    entities = set(mentions.values())
    relations = {(mentions[t.source_mention_id], normalize(t.relationship_type), mentions[t.target_mention_id])
                 for t in result.triples}
    return entities, relations


def metrics(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def score(gold: ExtractionResult, prediction: ExtractionResult | None) -> dict:
    reference = extraction_sets(gold)
    actual = extraction_sets(prediction)
    output = {}
    for name, expected, predicted in zip(("entities", "relations"), reference, actual):
        output[name] = {**metrics(len(expected & predicted), len(predicted - expected), len(expected - predicted)),
                        "missing": sorted(expected - predicted), "unexpected": sorted(predicted - expected)}
    return output


def aggregate(scores: list[dict]) -> dict:
    return {name: metrics(*(sum(s[name][count] for s in scores) for count in ("tp", "fp", "fn")))
            for name in ("entities", "relations")}
