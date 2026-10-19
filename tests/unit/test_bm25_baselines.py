"""Unit tests for free BM25 keyword baselines."""

import json
from pathlib import Path

import pytest

from eval.baselines.bm25 import (
    compute_support_f1,
    evaluate_per_question_bm25,
    evaluate_pooled_bm25,
    evaluate_random_baseline,
    tokenize,
)


def test_tokenize_basic():
    text = "Who is the CEO of Apple Inc. in 2020?"
    tokens = tokenize(text, remove_stopwords=True)
    assert "ceo" in tokens
    assert "apple" in tokens
    assert "inc" in tokens
    assert "2020" in tokens
    # Common stopwords removed
    assert "who" not in tokens
    assert "is" not in tokens
    assert "the" not in tokens
    assert "of" not in tokens
    assert "in" not in tokens


def test_tokenize_all_stopwords():
    tokens = tokenize("in on at to", remove_stopwords=True)
    # If all tokens are stopwords, returns original words to prevent empty queries
    assert len(tokens) > 0


def test_compute_support_f1():
    # Perfect match
    assert compute_support_f1({1, 2}, {1, 2}) == 1.0
    # No match
    assert compute_support_f1({1, 2}, {3, 4}) == 0.0
    # Partial match: gold={1, 2}, pred={1, 3, 4, 5, 6} (intersection=1, P=1/5=0.2, R=1/2=0.5 -> F1 = 2*(0.1)/0.7 = 0.2857)
    f1 = compute_support_f1({1, 2}, {1, 3, 4, 5, 6})
    assert pytest.approx(f1, 0.001) == 2 * (0.2 * 0.5) / (0.2 + 0.5)
    # Empty sets
    assert compute_support_f1(set(), {1, 2}) == 0.0
    assert compute_support_f1({1, 2}, set()) == 0.0


def test_evaluate_per_question_bm25_synthetic():
    records = [
        {
            "id": "q1",
            "question": "Where was Alan Turing born?",
            "paragraphs": [
                {"idx": 0, "title": "Alan Turing", "paragraph_text": "Alan Turing was born in Maida Vale, London.", "is_supporting": True},
                {"idx": 1, "title": "Maida Vale", "paragraph_text": "Maida Vale is a residential district in London.", "is_supporting": True},
                {"idx": 2, "title": "London Bus", "paragraph_text": "London buses are red double-deckers.", "is_supporting": False},
                {"idx": 3, "title": "Computer Science", "paragraph_text": "Computer science is the study of computation.", "is_supporting": False},
            ],
        }
    ]

    res = evaluate_per_question_bm25(records, k_values=[1, 2])
    assert res["setting"] == "per_question"
    assert res["question_count"] == 1
    # Both supporting paragraphs contain query terms "Alan Turing" / "born" / "London"
    assert res["recall_at_2"] > 0
    assert "hop_1" in res["per_hop_recall_at_5"]
    assert "hop_2" in res["per_hop_recall_at_5"]


def test_evaluate_pooled_bm25_synthetic():
    records = [
        {
            "id": "q1",
            "question": "Where was Alan Turing born?",
            "paragraphs": [
                {"idx": 0, "title": "Alan Turing", "paragraph_text": "Alan Turing was born in Maida Vale, London.", "is_supporting": True},
                {"idx": 1, "title": "Maida Vale", "paragraph_text": "Maida Vale is a district in London.", "is_supporting": True},
                {"idx": 2, "title": "Cooking Pasta", "paragraph_text": "Boil water and add salt before adding pasta.", "is_supporting": False},
            ],
        },
        {
            "id": "q2",
            "question": "How to make pasta?",
            "paragraphs": [
                {"idx": 0, "title": "Cooking Pasta", "paragraph_text": "Boil water and add salt before adding pasta.", "is_supporting": True},
                {"idx": 1, "title": "Alan Turing", "paragraph_text": "Alan Turing was born in Maida Vale, London.", "is_supporting": False},
            ],
        },
    ]

    res = evaluate_pooled_bm25(records, k_values=[2, 5])
    assert res["setting"] == "pooled"
    assert res["question_count"] == 2
    assert res["corpus_passages"] == 3  # Unique passages across all questions
    assert res["recall_at_2"] > 0
    assert res["recall_at_5"] >= res["recall_at_2"]


def test_random_baseline():
    records = [
        {
            "id": f"q_{i}",
            "paragraphs": [
                {"idx": j, "title": f"T{j}", "paragraph_text": f"P{j}", "is_supporting": (j < 2)}
                for j in range(20)
            ],
        }
        for i in range(100)
    ]

    res = evaluate_random_baseline(records, k=5, seed=42)
    assert res["setting"] == "random_per_question"
    assert res["question_count"] == 100
    # Expected recall when picking 5 of 20 paragraphs with 2 gold items:
    # Hypergeometric distribution: each gold item has 5/20 = 25% chance of being picked.
    # Mean recall should be approximately 25%
    assert 20.0 <= res["recall_at_k"] <= 30.0


def test_bm25_baseline_replicates_audit_benchmark():
    """Verify that BM25 reproduces audit benchmark protocol within ±1.0 percentage points.

    Audit findings (1,000 sampled questions from dev with seed 0):
      - Pooled BM25: R@2 ~ 33.9%, R@5 ~ 44.0%
      - Per-question BM25: R@5 ~ 49.5%
      - Random-5 baseline: ~ 25%
    """
    dev_path = Path("dataset/musique/data/data/musique_ans_v1.0_dev.jsonl")
    if not dev_path.exists():
        pytest.skip("Full dev dataset not present")

    import random

    with dev_path.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    rng = random.Random(0)
    sample_records = rng.sample(records, 1000)

    # Per-question BM25
    per_q = evaluate_per_question_bm25(sample_records, k_values=[2, 5])
    # Audit benchmark: 49.5% ± 1.0%
    assert abs(per_q["recall_at_5"] - 49.5) <= 1.0

    # Random-5 floor
    rand = evaluate_random_baseline(sample_records, k=5, seed=0)
    assert abs(rand["recall_at_k"] - 25.0) <= 1.5

    # Pooled BM25
    pooled = evaluate_pooled_bm25(sample_records, k_values=[2, 5])
    # Audit benchmark: R@2 ~ 33.9% ± 1.0%, R@5 ~ 44.0% ± 1.0%
    assert abs(pooled["recall_at_2"] - 33.9) <= 1.0
    assert abs(pooled["recall_at_5"] - 44.0) <= 1.0

    # Per-hop check: hop 1 should have much higher recall than hop 2 / hop 3
    hop_1 = pooled["per_hop_recall_at_5"]["hop_1"]
    hop_2 = pooled["per_hop_recall_at_5"]["hop_2"]
    assert hop_1 > hop_2, f"Hop 1 recall ({hop_1}%) should exceed Hop 2 recall ({hop_2}%)"
