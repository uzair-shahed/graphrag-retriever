"""Unit tests for the local model gateway (tools/model_gateway.py)."""

from __future__ import annotations

import json
import math
from pathlib import Path

from tools.model_gateway import (
    canonical_request_hash,
    compute_stub_embedding,
    extract_input_text,
    stub_answer_response,
    stub_graph_response,
    stub_schema_response,
)


def test_compute_stub_embedding_dimensions_and_norm() -> None:
    text = "Aster Systems acquired Northwind Labs in 2021"
    vec = compute_stub_embedding(text)
    assert len(vec) == 1536
    norm = math.sqrt(sum(v * v for v in vec))
    assert abs(norm - 1.0) < 1e-5


def test_canonical_request_hash_key_order_invariant() -> None:
    body1 = {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}], "temperature": 0}
    body2 = {"temperature": 0, "model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}]}
    assert canonical_request_hash(body1) == canonical_request_hash(body2)


def test_stub_schema_response_structure() -> None:
    schema = stub_schema_response()
    assert "node_types" in schema
    assert "relationship_types" in schema
    labels = {item["label"] for item in schema["node_types"]}
    assert "Company" in labels
    assert "Entity" in labels


def test_stub_graph_response_acquisition_pattern() -> None:
    text = "Aster Systems acquired Omega Corp in 2021"
    graph = stub_graph_response(text)
    assert len(graph["nodes"]) == 2
    assert graph["nodes"][0]["properties"]["name"] == "Aster Systems"
    assert graph["nodes"][1]["properties"]["name"] == "Omega Corp"
    assert graph["relationships"][0]["type"] == "ACQUIRED"
    assert graph["relationships"][0]["properties"]["since"] == "2021"


def test_stub_answer_response_citations() -> None:
    prompt = "Question: Where is Aster Systems?\nEvidence:\n[chunk-001] Aster Systems is in Rochester.\n"
    res_str = stub_answer_response(prompt)
    data = json.loads(res_str)
    assert "answer" in data
    assert "citations" in data
    assert "chunk-001" in data["citations"]


def test_extract_input_text_fallback() -> None:
    raw = "Extract entities.\nInput text:\nParagraph about Paris."
    assert extract_input_text(raw) == "Paragraph about Paris."
    plain = "Just plain text."
    assert extract_input_text(plain) == "Just plain text."
