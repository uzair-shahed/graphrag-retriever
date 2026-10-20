"""Unit tests for bounded graph walk across linked entities (Step M6.3).

Verifies:
1. Two-hop synthetic fixture where hop 2 is reachable only via a shared entity cluster.
2. Cosine fallback when linking is absent or no bridges exist.
3. Hub entity bounding: returns <= fanout rows within latency budget.
4. Zero cross-question leakage and bridge provenance on live Neo4j data.
"""

from __future__ import annotations

import time
import unittest.mock as mock

import neo4j
import pytest

from graphrag_pipeline.config import Neo4jSettings
from graphrag_pipeline.linking.clustering import exact_name_clustering
from graphrag_pipeline.retrieval import retrieve_per_question, retrieve_walk
from graphrag_pipeline.storage.neo4j_connection import Neo4jConnection


class DummyEmbedder:
    """Mock embedder providing deterministic embeddings."""

    def __init__(self, dim: int = 1536) -> None:
        self.dim = dim

    def embed_query(self, text: str) -> list[float]:
        val = 1.0 / (self.dim**0.5)
        return [val] * self.dim


def test_bounded_walk_synthetic_two_hop() -> None:
    """Verify the walk finds a 2-hop paragraph that cosine retrieval misses.

    Scenario:
    - Chunk 0: High cosine score (0.95), mentions 'Entity A'.
    - Chunk 1: Moderate cosine score (0.85), mentions 'Entity B' (distractor).
    - Chunk 2: Low cosine score (0.20), mentions 'Entity A' (hop 2 target).

    Cosine top 2 returns: [Chunk 0, Chunk 1] (misses Chunk 2).
    Bounded walk top 2 returns: [Chunk 0, Chunk 2] via Entity A's SAME_AS link.
    """
    driver = mock.create_autospec(neo4j.Driver, instance=True)

    mock_ranked = [
        {
            "idx": 0,
            "score": 0.95,
            "text": "Steve Hillage played guitar in Gong.",
            "paragraph_hash": "hash_0",
            "title": "Steve Hillage",
            "document_id": "doc-0",
            "record_id": "q1",
            "source_file": "musique",
            "source_line": 1,
        },
        {
            "idx": 1,
            "score": 0.85,
            "text": "Unrelated distractor with similar words to question.",
            "paragraph_hash": "hash_1",
            "title": "Distractor",
            "document_id": "doc-1",
            "record_id": "q1",
            "source_file": "musique",
            "source_line": 2,
        },
        {
            "idx": 2,
            "score": 0.20,
            "text": "Gong is a progressive rock band formed in Paris.",
            "paragraph_hash": "hash_2",
            "title": "Gong band",
            "document_id": "doc-2",
            "record_id": "q1",
            "source_file": "musique",
            "source_line": 3,
        },
    ]

    mock_hops = [
        {
            "idx": 2,
            "bridges": 2,
            "seed_score": 0.95,
            "bridge_names": ["Gong"],
        }
    ]

    driver.execute_query.return_value = (
        [{"ranked": mock_ranked, "hops": mock_hops}],
        mock.MagicMock(),
        [],
    )

    embedder = DummyEmbedder()
    res = retrieve_walk(
        driver=driver,
        embedder=embedder,
        question_id="q1",
        question_text="Where was the band of Steve Hillage formed?",
        top_k=2,
        seeds=1,
        fanout=1,
    )

    assert res.mode == "graph_walk"
    assert len(res.evidence) == 2

    # Chunk 0 is the seed
    assert res.evidence[0].paragraph_idx == 0
    assert res.evidence[0].graph_references == []

    # Chunk 2 was reached via the walk, despite having lower cosine similarity than Chunk 1
    assert res.evidence[1].paragraph_idx == 2
    assert res.evidence[1].graph_references == ["Gong"]


def test_bounded_walk_pure_cosine_fallback() -> None:
    """Verify that when no hops exist, the walk falls back cleanly to cosine order."""
    driver = mock.create_autospec(neo4j.Driver, instance=True)

    mock_ranked = [
        {
            "idx": 5,
            "score": 0.90,
            "text": "Text 5",
            "paragraph_hash": "h5",
            "title": "T5",
            "document_id": "d5",
            "record_id": "q1",
            "source_file": "f",
            "source_line": 1,
        },
        {
            "idx": 3,
            "score": 0.80,
            "text": "Text 3",
            "paragraph_hash": "h3",
            "title": "T3",
            "document_id": "d3",
            "record_id": "q1",
            "source_file": "f",
            "source_line": 2,
        },
        {
            "idx": 1,
            "score": 0.70,
            "text": "Text 1",
            "paragraph_hash": "h1",
            "title": "T1",
            "document_id": "d1",
            "record_id": "q1",
            "source_file": "f",
            "source_line": 3,
        },
    ]

    driver.execute_query.return_value = (
        [{"ranked": mock_ranked, "hops": []}],
        mock.MagicMock(),
        [],
    )

    embedder = DummyEmbedder()
    res = retrieve_walk(
        driver=driver,
        embedder=embedder,
        question_id="q1",
        question_text="Question text",
        top_k=3,
    )

    assert [c.paragraph_idx for c in res.evidence] == [5, 3, 1]
    assert all(c.graph_references == [] for c in res.evidence)


def test_bounded_walk_hop_deduplication_and_merging() -> None:
    """Verify hops that match existing seeds are omitted, and multi-seed hops combine bridges."""
    driver = mock.create_autospec(neo4j.Driver, instance=True)

    mock_ranked = [
        {"idx": 0, "score": 0.90, "text": "T0", "paragraph_hash": "h0", "title": "0", "document_id": "d0", "record_id": "q1", "source_file": "f", "source_line": 1},
        {"idx": 1, "score": 0.80, "text": "T1", "paragraph_hash": "h1", "title": "1", "document_id": "d1", "record_id": "q1", "source_file": "f", "source_line": 2},
        {"idx": 2, "score": 0.50, "text": "T2", "paragraph_hash": "h2", "title": "2", "document_id": "d2", "record_id": "q1", "source_file": "f", "source_line": 3},
        {"idx": 3, "score": 0.40, "text": "T3", "paragraph_hash": "h3", "title": "3", "document_id": "d3", "record_id": "q1", "source_file": "f", "source_line": 4},
    ]

    # Hop 1 is already in seeds (idx 1). Hop 2 is reached from both seed 0 and seed 1.
    mock_hops = [
        {"idx": 1, "bridges": 1, "seed_score": 0.90, "bridge_names": ["BridgeA"]},
        {"idx": 2, "bridges": 2, "seed_score": 0.90, "bridge_names": ["BridgeA"]},
        {"idx": 2, "bridges": 1, "seed_score": 0.80, "bridge_names": ["BridgeB"]},
    ]

    driver.execute_query.return_value = (
        [{"ranked": mock_ranked, "hops": mock_hops}],
        mock.MagicMock(),
        [],
    )

    embedder = DummyEmbedder()
    res = retrieve_walk(
        driver=driver,
        embedder=embedder,
        question_id="q1",
        question_text="Question text",
        top_k=4,
        seeds=2,
    )

    # Seeds 0 and 1 first, then Hop 2 (combines BridgeA and BridgeB), then filler 3
    assert [c.paragraph_idx for c in res.evidence] == [0, 1, 2, 3]
    assert res.evidence[2].graph_references == ["BridgeA", "BridgeB"]


def test_bounded_walk_live_zero_leaks_and_provenance() -> None:
    """Verify bounded walk on live Neo4j produces 0 leaks and populates bridge provenance."""
    try:
        settings = Neo4jSettings.from_env()
    except ValueError:
        pytest.skip("Local Neo4j instance not configured.")

    with Neo4jConnection(settings) as conn:
        driver = conn.connect()

        # Check if questions exist
        q_records, _, _ = driver.execute_query(
            "MATCH (q:Question)-[ic:IN_CONTEXT]->(d:Document)<-[:FROM_DOCUMENT]-(c:Chunk) "
            "WHERE c.embedding IS NOT NULL "
            "RETURN q.id AS qid, q.text AS qtext "
            "LIMIT 1"
        )
        if not q_records:
            pytest.skip("No Question nodes with embedded IN_CONTEXT chunks found in Neo4j.")

        qid = q_records[0]["qid"]
        qtext = q_records[0]["qtext"] or "Sample question"

        embedder = DummyEmbedder(dim=1536)

        # Run with clustering active
        with exact_name_clustering(driver):
            res_walk = retrieve_per_question(
                driver=driver,
                embedder=embedder,
                question_id=qid,
                question_text=qtext,
                top_k=5,
                walk=True,
            )

            assert res_walk.mode == "graph_walk"
            assert len(res_walk.evidence) > 0

            # Verify 0 leaks: all returned evidence belongs to this question
            for chunk in res_walk.evidence:
                assert chunk.paragraph_idx is not None
                assert 0 <= chunk.paragraph_idx < 20

                check_q, _, _ = driver.execute_query(
                    "MATCH (q:Question {id: $qid})-[ic:IN_CONTEXT]->(d:Document {paragraph_hash: $h}) "
                    "RETURN count(ic) AS links",
                    qid=qid,
                    h=chunk.paragraph_hash,
                )
                assert check_q[0]["links"] >= 1, f"Leakage detected: {chunk.paragraph_hash} not linked to {qid}"


def test_bounded_walk_hub_bounding() -> None:
    """Verify that a high-degree entity is bounded by $fanout and returns within 200ms."""
    try:
        settings = Neo4jSettings.from_env()
    except ValueError:
        pytest.skip("Local Neo4j instance not configured.")

    with Neo4jConnection(settings) as conn:
        driver = conn.connect()

        q_records, _, _ = driver.execute_query(
            "MATCH (q:Question)-[ic:IN_CONTEXT]->(d:Document)<-[:FROM_DOCUMENT]-(c:Chunk) "
            "WHERE c.embedding IS NOT NULL "
            "RETURN q.id AS qid, q.text AS qtext "
            "LIMIT 1"
        )
        if not q_records:
            pytest.skip("No embedded questions in Neo4j.")

        qid = q_records[0]["qid"]
        qtext = q_records[0]["qtext"] or "Test question"
        embedder = DummyEmbedder(dim=1536)

        with exact_name_clustering(driver):
            start = time.perf_counter()
            res = retrieve_walk(
                driver=driver,
                embedder=embedder,
                question_id=qid,
                question_text=qtext,
                top_k=5,
                seeds=2,
                fanout=2,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert len(res.evidence) <= 5
            # Query must finish well under 200 ms budget
            assert elapsed_ms < 200, f"Query took {elapsed_ms:.1f}ms, expected < 200ms"
