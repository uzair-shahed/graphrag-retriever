"""Live integration test for Question nodes, IN_CONTEXT links, and gold label leakage prevention."""

import json
import os
from pathlib import Path

import pytest
from neo4j.exceptions import ServiceUnavailable

from graphrag_pipeline.config import Neo4jSettings
from graphrag_pipeline.ingestion.record_adapter import (
    build_context_links_for_record,
    compute_paragraph_hash,
)
from graphrag_pipeline.storage.neo4j_connection import Neo4jConnection
from graphrag_pipeline.storage.question_context import (
    get_question_candidate_hashes,
    link_questions_to_paragraphs,
    verify_no_leakage_in_graph,
)


def _get_live_connection():
    os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
    os.environ.setdefault("NEO4J_USERNAME", "neo4j")
    os.environ.setdefault("NEO4J_PASSWORD", "changeme")
    os.environ.setdefault("NEO4J_DATABASE", "neo4j")

    try:
        settings = Neo4jSettings.from_env()
    except ValueError as e:
        pytest.skip(f"Neo4j not configured: {e}")

    conn = Neo4jConnection(settings)
    try:
        conn.verify_connectivity()
    except ServiceUnavailable:
        pytest.skip("Neo4j not reachable")
    return conn


@pytest.mark.live
def test_question_context_linking_and_no_leakage_on_real_data():
    """Verify linking 60 real MuSiQue questions to Documents with zero gold label leakage."""
    from tests.live.musique_ci import musique_inputs
    dataset_path, _slice_path = musique_inputs()

    # Read 60 questions
    with dataset_path.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for i, line in enumerate(f) if i < 60 and line.strip()]

    all_links = []
    unique_docs: dict[str, str] = {}  # hash -> title
    for r in records:
        raw_line = json.dumps(r)
        links = build_context_links_for_record(raw_line, source_line=1)
        all_links.extend(links)
        for p in r.get("paragraphs", []):
            h = compute_paragraph_hash(p["title"], p["paragraph_text"])
            unique_docs[h] = p["title"]

    conn = _get_live_connection()
    driver = conn.connect()

    try:
        # Clear graph for clean measurement
        with driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")

        # Create the Document nodes in batch
        doc_rows = [{"h": h, "title": t} for h, t in unique_docs.items()]
        with driver.session() as s:
            s.run(
                "UNWIND $rows AS r CREATE (:Document {paragraph_hash: r.h, title: r.title, ingest_status: 'complete'})",
                {"rows": doc_rows},
            )

        # Link questions to paragraphs
        links_created = link_questions_to_paragraphs(driver, all_links, batch_size=50)

        # Verify link count matches total candidate occurrences
        assert links_created == len(all_links)
        paragraphs_per_question = len(records[0]["paragraphs"])

        # Verify Question node count
        with driver.session() as s:
            q_count = s.run("MATCH (q:Question) RETURN count(q) AS c").single()["c"]
            d_count = s.run("MATCH (d:Document) RETURN count(d) AS c").single()["c"]
            r_count = s.run("MATCH ()-[r:IN_CONTEXT]->() RETURN count(r) AS c").single()["c"]

        assert q_count == len(records)
        assert d_count == len(unique_docs)
        assert r_count == len(all_links)

        # Verify each question has exactly 20 ordered candidate hashes
        first_q = records[0]["id"]
        candidates = get_question_candidate_hashes(driver, first_q)
        assert len(candidates) == paragraphs_per_question
        assert [idx for idx, _ in candidates] == list(range(paragraphs_per_question))

        # CRITICAL SAFETY CHECK: Zero gold labels in the graph
        verify_no_leakage_in_graph(driver)

        # Clean up
        with driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")

    finally:
        conn.close()
