"""Unit tests for per-question retrieval (Step M6.1).

Verifies exact cosine similarity over a question's IN_CONTEXT paragraphs with
zero leakage by construction.
"""

from __future__ import annotations

import unittest.mock as mock

import neo4j
import pytest

from graphrag_pipeline.config import Neo4jSettings
from graphrag_pipeline.retrieval import retrieve_per_question
from graphrag_pipeline.storage.neo4j_connection import Neo4jConnection


class DummyEmbedder:
    """Mock embedder providing deterministic embeddings."""

    def __init__(self, dim: int = 1536) -> None:
        self.dim = dim

    def embed_query(self, text: str) -> list[float]:
        # Return a deterministic unit vector
        val = 1.0 / (self.dim**0.5)
        return [val] * self.dim


def test_retrieve_per_question_mock_driver() -> None:
    """Verify retrieve_per_question queries IN_CONTEXT chunks and maps EvidenceChunks correctly."""
    driver = mock.create_autospec(neo4j.Driver, instance=True)

    mock_records = [
        {
            "text": "Steve Hillage is a British musician.",
            "paragraph_hash": "hash_001",
            "title": "Steve Hillage",
            "document_id": "doc-01",
            "record_id": "2hop__460946_294723",
            "source_file": "dataset/musique",
            "source_line": "1",
            "idx": 10,
            "score": 0.895,
        },
        {
            "text": "Grant Green recorded on Blue Note.",
            "paragraph_hash": "hash_002",
            "title": "Grant Green",
            "document_id": "doc-02",
            "record_id": "2hop__460946_294723",
            "source_file": "dataset/musique",
            "source_line": "1",
            "idx": 0,
            "score": 0.742,
        },
    ]

    driver.execute_query.return_value = (mock_records, mock.MagicMock(), [])

    embedder = DummyEmbedder()
    res = retrieve_per_question(
        driver=driver,
        embedder=embedder,
        question_id="2hop__460946_294723",
        question_text="Who is the spouse of the Green performer?",
        top_k=2,
    )

    assert res.mode == "per_question"
    assert res.question == "Who is the spouse of the Green performer?"
    assert len(res.evidence) == 2

    e0 = res.evidence[0]
    assert e0.chunk_id == "musique:paragraph:hash_001"
    assert e0.paragraph_hash == "hash_001"
    assert e0.paragraph_idx == 10
    assert e0.score == 0.895
    assert e0.source_line == 1

    e1 = res.evidence[1]
    assert e1.chunk_id == "musique:paragraph:hash_002"
    assert e1.paragraph_idx == 0
    assert e1.score == 0.742


def test_retrieve_per_question_invalid_embedder() -> None:
    """Verify ValueError is raised if embedder has no embed_query or get_embeddings."""
    driver = mock.create_autospec(neo4j.Driver, instance=True)
    with pytest.raises(ValueError, match="does not provide embed_query"):
        retrieve_per_question(
            driver=driver,
            embedder=object(),
            question_id="qid-1",
            question_text="Sample question?",
        )


def test_retrieve_per_question_live_zero_leakage() -> None:
    """Verify live per-question retrieval returns zero leakage: all results belong to the question's context."""
    try:
        settings = Neo4jSettings.from_env()
    except ValueError:
        pytest.skip("Local Neo4j instance not configured.")

    with Neo4jConnection(settings) as conn:
        driver = conn.connect()
        # Find a question with IN_CONTEXT links in the DB
        q_records, _, _ = driver.execute_query(
            "MATCH (q:Question)-[ic:IN_CONTEXT]->(d:Document)<-[:FROM_DOCUMENT]-(c:Chunk) "
            "WHERE c.embedding IS NOT NULL "
            "RETURN q.id AS qid, q.text AS qtext, count(c) AS chunk_count "
            "LIMIT 1"
        )
        if not q_records:
            pytest.skip("No Question nodes with embedded IN_CONTEXT chunks found in Neo4j.")

        qid = q_records[0]["qid"]
        qtext = q_records[0]["qtext"] or "Sample question"

        # Embed query using mock or unit vector matching chunk dimensions
        dim = 1536
        embedder = DummyEmbedder(dim=dim)

        res = retrieve_per_question(
            driver=driver,
            embedder=embedder,
            question_id=qid,
            question_text=qtext,
            top_k=5,
        )

        assert len(res.evidence) > 0
        # Check every evidence chunk has zero leakage
        for chunk in res.evidence:
            assert chunk.paragraph_idx is not None
            assert 0 <= chunk.paragraph_idx < 20

            # Verify in Neo4j that this paragraph is strictly connected to this question
            check_q, _, _ = driver.execute_query(
                "MATCH (q:Question {id: $qid})-[ic:IN_CONTEXT]->(d:Document {paragraph_hash: $h}) "
                "RETURN count(ic) AS links",
                qid=qid,
                h=chunk.paragraph_hash,
            )
            assert check_q[0]["links"] >= 1, f"Leakage detected! {chunk.paragraph_hash} not linked to {qid}"
