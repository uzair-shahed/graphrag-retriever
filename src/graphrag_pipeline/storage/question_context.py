"""Question nodes and IN_CONTEXT candidate paragraph links in Neo4j.

Links (:Question)-[:IN_CONTEXT {idx: paragraph_idx}]->(:Document {paragraph_hash})
without writing any gold labels (answers, is_supporting, question_decomposition).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Sequence

from neo4j import Driver

from graphrag_pipeline.ingestion.record_adapter import QuestionContextLink

logger = logging.getLogger(__name__)

FORBIDDEN_PROPERTIES = ("is_supporting", "answer", "answer_aliases", "question_decomposition")


class GraphLeakageError(RuntimeError):
    """Raised when forbidden gold labels are detected in the graph."""


def ensure_question_constraints(driver: Driver) -> None:
    """Ensure the unique constraint on Question.id exists."""
    query = (
        "CREATE CONSTRAINT question_id_unique IF NOT EXISTS "
        "FOR (q:Question) REQUIRE q.id IS UNIQUE"
    )
    with driver.session() as session:
        session.run(query).consume()


def link_questions_to_paragraphs(
    driver: Driver,
    links: Sequence[QuestionContextLink],
    batch_size: int = 100,
) -> int:
    """Create Question nodes and IN_CONTEXT relationships to Document nodes.

    Args:
        driver: Neo4j driver.
        links: Sequence of QuestionContextLink objects.
        batch_size: Number of questions to link per transaction.

    Returns:
        Total number of IN_CONTEXT links established.
    """
    ensure_question_constraints(driver)
    if not links:
        return 0

    # Group links by question
    by_question: dict[str, dict[str, object]] = {}
    for link in links:
        key = link.question_key
        if key not in by_question:
            by_question[key] = {
                "key": key,
                "text": link.question_text,
                "links": [],
            }
        by_question[key]["links"].append(  # type: ignore[union-attr]
            {"idx": link.paragraph_idx, "h": link.paragraph_hash}
        )

    rows = list(by_question.values())
    total_linked = 0

    query = """
    UNWIND $rows AS row
    MERGE (q:Question {id: row.key})
    SET q.text = row.text
    WITH q, row
    UNWIND row.links AS l
    MATCH (d:Document {paragraph_hash: l.h})
    MERGE (q)-[r:IN_CONTEXT {idx: l.idx}]->(d)
    RETURN count(r) AS links_created
    """

    with driver.session() as session:
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            result = session.run(query, {"rows": batch})
            record = result.single()
            if record:
                total_linked += record["links_created"]

    return total_linked


def verify_no_leakage_in_graph(driver: Driver) -> None:
    """Verify that zero gold labels exist as node or relationship properties in Neo4j."""
    query = "CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey"
    with driver.session() as session:
        result = session.run(query)
        keys = {record["propertyKey"].lower() for record in result}

    for forbidden in FORBIDDEN_PROPERTIES:
        if forbidden.lower() in keys:
            raise GraphLeakageError(
                f"Forbidden gold label {forbidden!r} detected in Neo4j property keys: {keys}"
            )


def get_question_candidate_hashes(
    driver: Driver, question_id: str
) -> list[tuple[int, str]]:
    """Retrieve ordered (idx, paragraph_hash) candidate pairs for a question."""
    query = """
    MATCH (q:Question {id: $qid})-[r:IN_CONTEXT]->(d:Document)
    RETURN r.idx AS idx, d.paragraph_hash AS paragraph_hash
    ORDER BY r.idx ASC
    """
    with driver.session() as session:
        result = session.run(query, {"qid": question_id})
        return [(record["idx"], record["paragraph_hash"]) for record in result]
