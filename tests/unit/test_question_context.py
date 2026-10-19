"""Unit tests for Question nodes, IN_CONTEXT links, and gold label leakage prevention."""

import ast
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from graphrag_pipeline.ingestion.record_adapter import QuestionContextLink
from graphrag_pipeline.storage.question_context import (
    GraphLeakageError,
    ensure_question_constraints,
    get_question_candidate_hashes,
    link_questions_to_paragraphs,
    verify_no_leakage_in_graph,
)


def test_ensure_question_constraints():
    mock_session = MagicMock()
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    ensure_question_constraints(mock_driver)

    mock_session.run.assert_called_once()
    query = mock_session.run.call_args[0][0]
    assert "CREATE CONSTRAINT question_id_unique IF NOT EXISTS" in query
    assert "FOR (q:Question) REQUIRE q.id IS UNIQUE" in query


def test_link_questions_to_paragraphs():
    mock_session = MagicMock()
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    mock_result = MagicMock()
    mock_result.single.return_value = {"links_created": 3}
    mock_session.run.return_value = mock_result

    links = [
        QuestionContextLink(
            question_key="q_1",
            question_text="Who founded Aster?",
            paragraph_idx=0,
            paragraph_hash="hash_a",
        ),
        QuestionContextLink(
            question_key="q_1",
            question_text="Who founded Aster?",
            paragraph_idx=1,
            paragraph_hash="hash_b",
        ),
        QuestionContextLink(
            question_key="q_2",
            question_text="Where is London?",
            paragraph_idx=0,
            paragraph_hash="hash_c",
        ),
    ]

    total = link_questions_to_paragraphs(mock_driver, links, batch_size=10)
    assert total == 3

    # Verify query structure
    args, kwargs = mock_session.run.call_args
    query = args[0]
    assert "MERGE (q:Question {id: row.key})" in query
    assert "MATCH (d:Document {paragraph_hash: l.h})" in query
    assert "MERGE (q)-[r:IN_CONTEXT {idx: l.idx}]->(d)" in query

    params = args[1]
    rows = params["rows"]
    assert len(rows) == 2  # 2 questions
    assert rows[0]["key"] == "q_1"
    assert len(rows[0]["links"]) == 2


def test_verify_no_leakage_in_graph_passes_when_clean():
    mock_session = MagicMock()
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    mock_session.run.return_value = [
        {"propertyKey": "text"},
        {"propertyKey": "title"},
        {"propertyKey": "paragraph_hash"},
        {"propertyKey": "ingest_status"},
        {"propertyKey": "idx"},
    ]

    # Should pass without error
    verify_no_leakage_in_graph(mock_driver)


def test_verify_no_leakage_in_graph_fails_when_gold_labels_present():
    mock_session = MagicMock()
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    mock_session.run.return_value = [
        {"propertyKey": "text"},
        {"propertyKey": "is_supporting"},  # Forbidden gold label
    ]

    with pytest.raises(GraphLeakageError, match="is_supporting"):
        verify_no_leakage_in_graph(mock_driver)


def test_get_question_candidate_hashes():
    mock_session = MagicMock()
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    mock_session.run.return_value = [
        {"idx": 0, "paragraph_hash": "hash_0"},
        {"idx": 1, "paragraph_hash": "hash_1"},
    ]

    candidates = get_question_candidate_hashes(mock_driver, "q_123")
    assert candidates == [(0, "hash_0"), (1, "hash_1")]
    args = mock_session.run.call_args[0]
    assert "MATCH (q:Question {id: $qid})-[r:IN_CONTEXT]->(d:Document)" in args[0]


def test_inference_code_never_parses_hop_count_from_id():
    """Verify that inference and retrieval modules never parse hop counts from question IDs.

    MuSiQue IDs have prefixes like '2hop__1_2'. Reading this prefix at inference
    would be target leakage.
    """
    src_dir = Path("src/graphrag_pipeline")
    inference_files = [
        src_dir / "retrieval.py",
        src_dir / "answer.py",
        src_dir / "query.py",
        src_dir / "query_command.py",
    ]

    forbidden_patterns = ["2hop", "3hop", "4hop", "r\"(\\d+)hop\"", "r'(\\d+)hop'"]

    for file_path in inference_files:
        if not file_path.exists():
            continue
        content = file_path.read_text(encoding="utf-8")
        for pattern in forbidden_patterns:
            assert pattern not in content, (
                f"Inference module {file_path.name} contains forbidden hop-parsing pattern {pattern!r}"
            )
