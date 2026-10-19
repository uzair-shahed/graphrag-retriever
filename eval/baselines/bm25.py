"""BM25 keyword baselines for MuSiQue in per-question and pooled settings.

Measures Recall@2, Recall@5, per-hop recall, and support F1.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further",
    "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself",
    "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "me", "more", "most",
    "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "our",
    "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such",
    "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "with", "you",
    "your", "yours", "yourself", "yourselves",
}


def tokenize(text: str, remove_stopwords: bool = False) -> list[str]:
    """Tokenize lowercased text into terms, optionally removing standard stopwords."""
    if remove_stopwords:
        words = re.findall(r"\w+", text.lower())
        filtered = [w for w in words if w not in STOPWORDS]
        return filtered if filtered else words
    return text.lower().split()


def get_hop_supporting_indices(record: dict[str, Any]) -> list[int]:
    """Extract ordered supporting paragraph indices for each reasoning hop.

    In MuSiQue, question_decomposition specifies the exact chain of hops with
    paragraph_support_idx for each hop. When absent, falls back to paragraphs
    with is_supporting=True in their original order.
    """
    decomp = record.get("question_decomposition", [])
    if decomp:
        return [step["paragraph_support_idx"] for step in decomp if "paragraph_support_idx" in step]

    paragraphs = record.get("paragraphs", [])
    return [p["idx"] for p in paragraphs if p.get("is_supporting")]


def compute_support_f1(gold_indices: set[int], retrieved_indices: set[int]) -> float:
    """Compute F1 overlap between gold supporting paragraph indices and retrieved indices."""
    if not gold_indices or not retrieved_indices:
        return 0.0
    intersection = len(gold_indices & retrieved_indices)
    precision = intersection / len(retrieved_indices)
    recall = intersection / len(gold_indices)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def evaluate_per_question_bm25(
    records: list[dict[str, Any]],
    k_values: list[int] = [2, 5],
    remove_stopwords: bool = False,
) -> dict[str, Any]:
    """Evaluate BM25 within each question's candidate paragraphs (per-question setting)."""
    recalls_by_k: dict[int, list[float]] = {k: [] for k in k_values}
    support_f1_scores: list[float] = []
    hop_recalls: dict[int, list[float]] = defaultdict(list)

    for record in records:
        paragraphs = record.get("paragraphs", [])
        if not paragraphs:
            continue
        supporting_indices = set(get_hop_supporting_indices(record))
        if not supporting_indices:
            continue

        corpus = [tokenize(f"{p['title']} {p['paragraph_text']}", remove_stopwords=remove_stopwords) for p in paragraphs]
        bm25 = BM25Okapi(corpus)
        q_tokens = tokenize(record["question"], remove_stopwords=remove_stopwords)
        scores = bm25.get_scores(q_tokens)

        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        ranked_p_indices = [paragraphs[i]["idx"] for i in sorted_indices]

        for k in k_values:
            top_k_indices = set(ranked_p_indices[:k])
            recall = len(supporting_indices & top_k_indices) / len(supporting_indices)
            recalls_by_k[k].append(recall)

        top_5_indices = set(ranked_p_indices[:5])
        support_f1_scores.append(compute_support_f1(supporting_indices, top_5_indices))

        hop_chain = get_hop_supporting_indices(record)
        for hop_num, supp_idx in enumerate(hop_chain, start=1):
            hop_recalls[hop_num].append(1.0 if supp_idx in top_5_indices else 0.0)

    results: dict[str, Any] = {
        "setting": "per_question",
        "question_count": len(records),
        "recall_at_2": round(sum(recalls_by_k[2]) / len(recalls_by_k[2]) * 100, 2) if recalls_by_k.get(2) else 0.0,
        "recall_at_5": round(sum(recalls_by_k[5]) / len(recalls_by_k[5]) * 100, 2) if recalls_by_k.get(5) else 0.0,
        "support_f1": round(sum(support_f1_scores) / len(support_f1_scores) * 100, 2) if support_f1_scores else 0.0,
        "per_hop_recall_at_5": {
            f"hop_{h}": round(sum(scores) / len(scores) * 100, 2)
            for h, scores in sorted(hop_recalls.items())
        },
    }
    return results


def evaluate_pooled_bm25(
    records: list[dict[str, Any]],
    k_values: list[int] = [2, 5],
    remove_stopwords: bool = True,
) -> dict[str, Any]:
    """Evaluate BM25 over all unique passages across the questions (pooled setting)."""
    passage_list: list[tuple[str, str]] = []
    seen_passages: set[tuple[str, str]] = set()

    for r in records:
        for p in r.get("paragraphs", []):
            key = (p["title"], p["paragraph_text"])
            if key not in seen_passages:
                seen_passages.add(key)
                passage_list.append(key)

    corpus = [tokenize(f"{title} {text}", remove_stopwords=remove_stopwords) for title, text in passage_list]
    bm25 = BM25Okapi(corpus)

    recalls_by_k: dict[int, list[float]] = {k: [] for k in k_values}
    support_f1_scores: list[float] = []
    hop_recalls: dict[int, list[float]] = defaultdict(list)

    for record in records:
        paragraphs = record.get("paragraphs", [])
        hop_indices = get_hop_supporting_indices(record)
        idx_to_passage = {p["idx"]: (p["title"], p["paragraph_text"]) for p in paragraphs}
        gold_passages = {idx_to_passage[idx] for idx in hop_indices if idx in idx_to_passage}
        if not gold_passages:
            continue

        q_tokens = tokenize(record["question"], remove_stopwords=remove_stopwords)
        scores = bm25.get_scores(q_tokens)
        max_k = max(k_values)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:max_k]

        ranked_passages = [passage_list[i] for i in top_indices]

        for k in k_values:
            top_k_passages = set(ranked_passages[:k])
            recall = len(gold_passages & top_k_passages) / len(gold_passages)
            recalls_by_k[k].append(recall)

        top_5_passages = set(ranked_passages[:5])
        intersection = len(gold_passages & top_5_passages)
        precision = intersection / 5
        recall = intersection / len(gold_passages)
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
        support_f1_scores.append(f1 * 100)

        for hop_num, idx in enumerate(hop_indices, start=1):
            if idx in idx_to_passage:
                key = idx_to_passage[idx]
                hop_recalls[hop_num].append(1.0 if key in top_5_passages else 0.0)

    results: dict[str, Any] = {
        "setting": "pooled",
        "question_count": len(records),
        "corpus_passages": len(passage_list),
        "recall_at_2": round(sum(recalls_by_k[2]) / len(recalls_by_k[2]) * 100, 2) if recalls_by_k.get(2) else 0.0,
        "recall_at_5": round(sum(recalls_by_k[5]) / len(recalls_by_k[5]) * 100, 2) if recalls_by_k.get(5) else 0.0,
        "support_f1": round(sum(support_f1_scores) / len(support_f1_scores), 2) if support_f1_scores else 0.0,
        "per_hop_recall_at_5": {
            f"hop_{h}": round(sum(scores) / len(scores) * 100, 2)
            for h, scores in sorted(hop_recalls.items())
        },
    }
    return results


def evaluate_random_baseline(
    records: list[dict[str, Any]],
    k: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    """Evaluate a random selection floor in the per-question setting."""
    rng = random.Random(seed)
    recalls: list[float] = []
    hop_recalls: dict[int, list[float]] = defaultdict(list)

    for record in records:
        paragraphs = record.get("paragraphs", [])
        hop_indices = get_hop_supporting_indices(record)
        supporting_indices = set(hop_indices)
        if not supporting_indices or len(paragraphs) < k:
            continue

        sampled = set(rng.sample([p["idx"] for p in paragraphs], k))
        recalls.append(len(supporting_indices & sampled) / len(supporting_indices))

        for hop_num, supp_idx in enumerate(hop_indices, start=1):
            hop_recalls[hop_num].append(1.0 if supp_idx in sampled else 0.0)

    return {
        "setting": "random_per_question",
        "k": k,
        "question_count": len(records),
        "recall_at_k": round(sum(recalls) / len(recalls) * 100, 2) if recalls else 0.0,
        "per_hop_recall_at_k": {
            f"hop_{h}": round(sum(scores) / len(scores) * 100, 2)
            for h, scores in sorted(hop_recalls.items())
        },
    }


def run_baselines_on_slice(
    slice_path: Path,
    split: str = "both",
) -> dict[str, Any]:
    """Run all baselines on a slice JSON (loading tune and/or test records)."""
    with slice_path.open("r", encoding="utf-8") as f:
        slice_meta = json.load(f)

    gold_path = slice_path.parent / f"{slice_path.stem}_gold.jsonl"
    if not gold_path.exists():
        gold_path = Path(slice_meta["source_file"])

    all_records = []
    with gold_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_records.append(json.loads(line.strip()))

    tune_ids = set(slice_meta.get("tune_ids", []))
    test_ids = set(slice_meta.get("test_ids", []))

    tune_records = [r for r in all_records if r["id"] in tune_ids]
    test_records = [r for r in all_records if r["id"] in test_ids]

    reports: dict[str, Any] = {"slice": slice_meta["name"]}

    if split in ("tune", "both"):
        reports["tune"] = {
            "per_question_bm25": evaluate_per_question_bm25(tune_records),
            "pooled_bm25": evaluate_pooled_bm25(tune_records),
            "random_5": evaluate_random_baseline(tune_records, k=5),
        }

    if split in ("test", "both"):
        reports["test"] = {
            "per_question_bm25": evaluate_per_question_bm25(test_records),
            "pooled_bm25": evaluate_pooled_bm25(test_records),
            "random_5": evaluate_random_baseline(test_records, k=5),
        }

    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", type=Path, default=Path("eval/slices/ans_dev_mvp_v1.json"))
    parser.add_argument("--split", choices=("tune", "test", "both"), default="both")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    results = run_baselines_on_slice(args.slice, split=args.split)
    output_str = json.dumps(results, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_str, encoding="utf-8")
        print(f"Wrote baseline report to {args.output}")
    else:
        print(output_str)


if __name__ == "__main__":
    main()
