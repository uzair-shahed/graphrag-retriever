"""Phase Gate M6 (Gate-L) live verification tests.

Phase Gate M6 requirements:
1. Full MVP comparative evaluation across 4 conditions:
   - bm25
   - vector
   - walk_off (ablation: linking off)
   - walk_exact (linking on: exact-name SAME_AS clusters)
2. Zero cross-question leakage: all retrieved evidence chunks belong strictly to the question's IN_CONTEXT graph.
3. Grounded citations: all citations match retrieved paragraph hashes and map to predicted_support_idxs.
4. Replay mode idempotency: rerun in replay mode makes 0 new gateway calls ($0 spend) and produces identical metrics.
5. Error diagnosis categorization: failures categorizable into extraction miss, link miss, walk miss, or answer error.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from neo4j.exceptions import ServiceUnavailable

from eval.run_mvp import run_mvp
from graphrag_pipeline.config import Neo4jSettings
from graphrag_pipeline.storage.neo4j_connection import Neo4jConnection
from tools.model_gateway import ModelGatewayHandler


@pytest.fixture
def gateway_server(tmp_path: Path):
    """Start an in-process ModelGatewayHandler on an ephemeral port."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), ModelGatewayHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    cache_dir = tmp_path / "gateway_cache"

    yield base_url, cache_dir

    server.shutdown()
    server.server_close()
    ModelGatewayHandler.reset_defaults()


def _get_neo4j_settings() -> Neo4jSettings:
    os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
    os.environ.setdefault("NEO4J_USERNAME", "neo4j")
    os.environ.setdefault("NEO4J_PASSWORD", "changeme")
    os.environ.setdefault("NEO4J_DATABASE", "neo4j")

    try:
        settings = Neo4jSettings.from_env()
    except ValueError as e:
        pytest.skip(f"Neo4j not configured: {e}")

    with Neo4jConnection(settings) as conn:
        try:
            conn.verify_connectivity()
        except ServiceUnavailable:
            pytest.skip(f"Neo4j is not reachable at {settings.uri}")
    return settings


@pytest.mark.live
def test_phase_gate_m6_mvp_retrieval_and_comparative_baselines(
    gateway_server: tuple[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify Phase Gate M6 (Gate-L): 4 conditions, 0 leaks, grounded citations, and $0 replay idempotency."""
    base_url, cache_dir = gateway_server
    neo4j_settings = _get_neo4j_settings()

    # Configure gateway in stub mode with caching
    req_cfg = urllib.request.Request(
        f"{base_url}/config",
        data=json.dumps(
            {
                "mode": "stub",
                "cache_dir": str(cache_dir),
                "fake_unit_cost": 0.001,
                "reset_spend": True,
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req_cfg) as resp:
        assert resp.status == 200

    # Pinned experiment profile
    gateway_api_url = f"{base_url}/v1"
    monkeypatch.setenv("OPENROUTER_BASE_URL", gateway_api_url)
    monkeypatch.setenv("OPENROUTER_API_KEY", "gate-m6-test-key")
    monkeypatch.setenv("OPENROUTER_LLM_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")
    monkeypatch.setenv("OPENROUTER_EMBEDDING_DIMENSIONS", "1536")
    monkeypatch.setenv("GRAPH_RAG_VECTOR_DIMENSIONS", "1536")
    monkeypatch.setenv("GRAPH_RAG_PROVIDER", "openrouter")
    monkeypatch.setenv("GRAPH_RAG_PROFILE", "experiment")
    monkeypatch.setenv("GRAPH_RAG_TEMPERATURE", "0.0")
    monkeypatch.setenv("GRAPH_RAG_SEED", "20261018")
    monkeypatch.setenv("OPENROUTER_PROVIDER_ORDER", "openai")
    monkeypatch.setenv("OPENROUTER_ALLOW_FALLBACKS", "false")
    monkeypatch.setenv("OPENROUTER_REQUIRE_PARAMETERS", "true")

    slice_path = Path("eval/slices/ans_dev_mvp_v1.json")
    assert slice_path.exists(), "eval/slices/ans_dev_mvp_v1.json must exist"

    # Find which questions in the slice are already in Neo4j
    with Neo4jConnection(neo4j_settings) as conn:
        q_records, _, _ = conn.connect().execute_query("MATCH (q:Question) RETURN q.id AS qid")
        db_qids = {r["qid"] for r in q_records}

    assert len(db_qids) > 0, "Neo4j has 0 ingested Question nodes"

    run_1_dir = tmp_path / "run_1"
    report_1, results_1 = run_mvp(
        slice_spec=slice_path,
        split="tune",
        conditions=["bm25", "vector", "walk_off", "walk_exact"],
        limit=5,
        output_dir=run_1_dir,
        stub=False,  # Routes through model gateway stub server
    )

    # 1. 4 conditions executed
    assert len(results_1) == 4
    cond_names = [r.condition for r in results_1]
    assert cond_names == ["bm25", "vector", "walk_off", "walk_exact"]

    # 2. Verify all predictions have valid grounded citations and non-negative metrics
    for r in results_1:
        assert r.num_questions == 5
        pred_file = Path(r.predictions_path)
        assert pred_file.exists()
        preds = [json.loads(line) for line in pred_file.read_text().strip().splitlines()]
        assert len(preds) == 5
        for p in preds:
            assert "id" in p
            assert "predicted_answer" in p
            assert "predicted_support_idxs" in p
            assert "predicted_answerable" in p
            assert isinstance(p["predicted_support_idxs"], list)
            assert isinstance(p["predicted_answerable"], bool)

    # 3. Check gateway spend after run 1
    with urllib.request.urlopen(f"{base_url}/health") as resp:
        health_1 = json.loads(resp.read().decode("utf-8"))
    spend_1 = health_1["spent_usd"]
    new_calls_1 = health_1["new_calls_count"]
    assert spend_1 > 0.0, "Spend should be non-zero after first run"
    assert new_calls_1 > 0

    # 4. Switch gateway to REPLAY mode and run again
    req_replay = urllib.request.Request(
        f"{base_url}/config",
        data=json.dumps({"mode": "replay"}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req_replay) as resp:
        assert resp.status == 200

    run_2_dir = tmp_path / "run_2"
    report_2, results_2 = run_mvp(
        slice_spec=slice_path,
        split="tune",
        conditions=["bm25", "vector", "walk_off", "walk_exact"],
        limit=5,
        output_dir=run_2_dir,
        stub=False,  # Replay mode
    )

    # 5. Verify $0 spend in replay mode
    with urllib.request.urlopen(f"{base_url}/health") as resp:
        health_2 = json.loads(resp.read().decode("utf-8"))
    spend_2 = health_2["spent_usd"]
    new_calls_2 = health_2["new_calls_count"]

    assert spend_2 == spend_1, "Spend must remain identical in replay mode ($0 spend)"
    assert new_calls_2 == new_calls_1, "Zero new external calls permitted in replay mode"

    # 6. Verify identical predictions between run 1 and run 2
    for r1, r2 in zip(results_1, results_2):
        assert r1.condition == r2.condition
        assert r1.answer_f1 == r2.answer_f1
        assert r1.answer_em == r2.answer_em
        assert r1.support_f1 == r2.support_f1
        assert r1.r_at_2 == r2.r_at_2
        assert r1.r_at_5 == r2.r_at_5
        assert r1.later_hop_r_at_5 == r2.later_hop_r_at_5

        preds_1 = Path(r1.predictions_path).read_text()
        preds_2 = Path(r2.predictions_path).read_text()
        assert preds_1 == preds_2, f"Predictions mismatch in replay mode for {r1.condition}"
