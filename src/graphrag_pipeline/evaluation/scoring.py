"""Gold-label metrics and fault-isolated RAGAS Triad scoring."""

from __future__ import annotations

import asyncio
import math
import re
import string
from collections import Counter

from graphrag_pipeline.answer_context import render_evidence_chunk
from graphrag_pipeline.diagnostics import exception_diagnostics


def normalize_answer(value: str) -> str:
    value = value.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", value).split())


def answer_f1(answer, reference):
    a, b = normalize_answer(answer).split(), normalize_answer(reference).split()
    if not a or not b:
        return float(a == b)
    shared = sum((Counter(a) & Counter(b)).values())
    return 2 * shared / (len(a) + len(b))


def evidence_metrics(case, evidence, citations=()):
    expected = {f"musique:{case.gold.id}:paragraph:{p.idx}": p for p in case.gold.paragraphs}
    # ID alone never proves support: require the unchanged gold paragraph text.
    matched = {e.chunk_id for e in evidence if e.chunk_id in expected and e.text == expected[e.chunk_id].paragraph_text}
    supporting = {k for k, p in expected.items() if p.is_supporting}
    covered = matched & supporting
    hop_checks = [
        {"hop_id": h.id, "paragraph_support_idx": h.paragraph_support_idx,
         "covered": f"musique:{case.gold.id}:paragraph:{h.paragraph_support_idx}" in matched if h.paragraph_support_idx is not None else None}
        for h in case.gold.question_decomposition
    ]
    known_hops = [h["covered"] for h in hop_checks if h["covered"] is not None]
    unique = {e.chunk_id for e in evidence}
    return {
        "supporting_recall": len(covered) / len(supporting) if supporting else None,
        "supporting_precision": len(covered) / len(unique) if unique else 0.0,
        "supporting_covered": sorted(covered),
        "all_supporting_covered": supporting <= matched if supporting else None,
        "hop_coverage": sum(known_hops) / len(known_hops) if known_hops else None,
        "all_hops_covered": all(known_hops) if known_hops else None,
        "hops": hop_checks,
        "citation_gold_support_precision": len(set(citations) & covered) / len(set(citations)) if citations else None,
        "citation_gold_support_recall": len(set(citations) & covered) / len(supporting) if supporting else None,
    }


def gold_metrics(case, response):
    answered = response.status == "succeeded"
    abstained = response.status == "insufficient_evidence"
    # Infrastructure failure must never count as a correct abstention.
    correct = False
    f1 = 0.0
    if case.gold.answerable:
        references = [case.gold.answer, *case.gold.answer_aliases]
        if answered:
            correct = any(normalize_answer(response.answer) == normalize_answer(r) for r in references)
            f1 = max(answer_f1(response.answer, r) for r in references)
    else:
        correct = abstained
        f1 = float(correct)
    return {
        "answer_correct": correct, "answer_f1": f1,
        "answerability_correct": (answered == case.gold.answerable) if response.status != "failed" else False,
        "gold_answerable": case.gold.answerable,
    }


async def triad_scores(metrics, response, evidence, *, timeout=180):
    contexts = [render_evidence_chunk(e) for e in evidence]
    results = {}
    for name, metric in metrics.items():
        if not contexts or (name != "context_relevance" and response.status != "succeeded"):
            results[name] = {"value": None, "status": "not_applicable"}
            continue
        kwargs = {"user_input": response.question}
        if name != "answer_relevancy":
            kwargs["retrieved_contexts"] = contexts
        if name != "context_relevance":
            kwargs["response"] = response.answer
        try:
            result = await asyncio.wait_for(metric.ascore(**kwargs), timeout)
            value = float(result.value)
            if not math.isfinite(value):
                raise ValueError("metric returned a non-finite score")
            results[name] = {"value": value, "status": "scored", "reason": getattr(result, "reason", None)}
        except Exception as error:
            results[name] = {"value": None, "status": "failed", "error": exception_diagnostics(error)}
    return results
