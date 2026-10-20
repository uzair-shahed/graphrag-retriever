"""Unit tests for graph quality and bridge coverage evaluation (Step M5.3)."""

from __future__ import annotations

import unittest.mock as mock
from pathlib import Path

import neo4j
import pytest

from eval.graph_quality import (
    GraphQualityReport,
    compute_graph_quality,
    extract_bridge_links,
    normalize_entity_name,
    render_graph_quality_report,
)
from graphrag_pipeline.config import Neo4jSettings
from graphrag_pipeline.storage.neo4j_connection import Neo4jConnection


def test_normalize_entity_name() -> None:
    """Verify name normalization strips leading articles, punctuation, and whitespace."""
    assert normalize_entity_name("  The United States  ") == "united states"
    assert normalize_entity_name("A Company, Inc.") == "company inc"
    assert normalize_entity_name("An Apple!") == "apple"
    assert normalize_entity_name("Steve Hillage") == "steve hillage"


def test_extract_bridge_links_multi_hop() -> None:
    """Verify bridge link extraction from MuSiQue question decompositions."""
    record = {
        "id": "2hop__test_01",
        "question_decomposition": [
            {
                "id": 101,
                "question": "Who directed the film?",
                "answer": "Christopher Nolan",
                "paragraph_support_idx": 4,
            },
            {
                "id": 102,
                "question": "Where was #1 born?",
                "answer": "London",
                "paragraph_support_idx": 8,
            },
        ],
    }

    bridges = extract_bridge_links(record)
    assert len(bridges) == 1
    b = bridges[0]
    assert b["bridge_entity"] == "Christopher Nolan"
    assert b["hop_source_idx"] == 4
    assert b["hop_target_idx"] == 8
    assert b["source_hop_id"] == 101
    assert b["target_hop_id"] == 102


def test_render_graph_quality_report() -> None:
    """Verify report rendering outputs structured markdown with tables and statistics."""
    report = GraphQualityReport(
        total_paragraphs=10,
        paragraphs_with_zero_entities=1,
        zero_entities_ratio=0.1,
        entities_per_paragraph={"mean": 3.5, "min": 0.0, "max": 7.0},
        type_mix={"Person": 15, "Location": 10, "Organization": 5},
        relation_histogram={"BORN_IN": 8, "LOCATED_IN": 7},
        largest_name_groups=[{"name": "London", "count": 4}],
        total_bridge_links=2,
        covered_bridge_links=1,
        bridge_coverage_ratio=0.5,
    )
    rendered = render_graph_quality_report(report)
    assert "# Graph Quality & Bridge Coverage Report" in rendered
    assert "**Total Paragraphs (Documents):** 10" in rendered
    assert "| Person | 15 | 50.0% |" in rendered
    assert "| BORN_IN | 8 |" in rendered
    assert "| London | 4 |" in rendered
    assert "**Bridge Links Covered:** 1 (50.0%)" in rendered


def test_compute_graph_quality_with_mock_driver() -> None:
    """Verify compute_graph_quality handles query execution and computes correct ratios."""
    driver = mock.create_autospec(neo4j.Driver, instance=True)

    # 1. doc_count_query
    doc_res = [
        {"hash": "h1", "entity_count": 3},
        {"hash": "h2", "entity_count": 0},
    ]
    # 2. type_query
    type_res = [{"lbl": "Concept", "count": 3}]
    # 3. rel_query
    rel_res = [{"rel_type": "RELATED_TO", "count": 2}]
    # 4. name_query
    name_res = [{"name": "Apollo", "count": 2}]
    # 5. entity_map_query
    map_res = [
        {"record_id": "rec-1", "paragraph_idx": "0", "entities": ["Apollo", "NASA"]},
        {"record_id": "rec-1", "paragraph_idx": "1", "entities": ["Apollo", "Moon"]},
    ]

    driver.execute_query.side_effect = [
        (doc_res, None, None),
        (type_res, None, None),
        (rel_res, None, None),
        (name_res, None, None),
        (map_res, None, None),
    ]

    records = [
        {
            "id": "rec-1",
            "question_decomposition": [
                {"id": 1, "question": "What mission?", "answer": "Apollo", "paragraph_support_idx": 0},
                {"id": 2, "question": "Where did #1 land?", "answer": "Moon", "paragraph_support_idx": 1},
            ],
        }
    ]

    report = compute_graph_quality(driver, records)
    assert report.total_paragraphs == 2
    assert report.paragraphs_with_zero_entities == 1
    assert report.zero_entities_ratio == 0.5
    assert report.total_bridge_links == 1
    assert report.covered_bridge_links == 1
    assert report.bridge_coverage_ratio == 1.0


def test_compute_graph_quality_live_neo4j() -> None:
    """Verify compute_graph_quality runs against live local Neo4j when available."""
    try:
        settings = Neo4jSettings.from_env()
    except ValueError:
        pytest.skip("Local Neo4j instance not configured.")

    with Neo4jConnection(settings) as conn:
        report = compute_graph_quality(conn.connect())
        assert report.total_paragraphs >= 0
        assert report.zero_entities_ratio >= 0.0
        assert isinstance(report.type_mix, dict)
        assert isinstance(report.relation_histogram, dict)
