"""Phase Gate M4 verification tests.

Phase gate M4 requirements:
1. Answer 10 slice questions from BM25 context with the experiment profile.
2. 10/10 valid JSON outputs, pinned provider (openai), costs logged.
3. Rerun in replay mode: $0 and identical outputs.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
from rank_bm25 import BM25Okapi

from eval.baselines.bm25 import tokenize
from graphrag_pipeline.answer import build_answer_llm, generate_answer
from graphrag_pipeline.config import OpenRouterSettings
from graphrag_pipeline.query import EvidenceChunk, QueryRequest
from graphrag_pipeline.retrieval import RetrievalResult
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


def test_phase_gate_m4_strict_json_and_replay_idempotency(
    gateway_server: tuple[str, Path],
) -> None:
    """Verify Phase Gate M4: 10 slice answers, pinned provider, and $0 replay mode."""
    base_url, cache_dir = gateway_server

    # Configure gateway in stub mode with caching enabled
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

    with urllib.request.urlopen(f"{base_url}/requests/clear") as resp:
        assert resp.status == 200

    # Load 10 slice questions
    slice_file = Path("eval/slices/ans_dev_mvp_v1.json")
    assert slice_file.exists(), "eval/slices/ans_dev_mvp_v1.json must exist"
    with slice_file.open("r", encoding="utf-8") as f:
        slice_data = json.load(f)
    target_ids = set(slice_data["tune_ids"][:10])

    dataset_file = Path("dataset/musique/data/data/musique_ans_v1.0_dev.jsonl")
    if not dataset_file.exists():
        pytest.skip("MuSiQue dataset file not downloaded locally.")

    records = []
    with dataset_file.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record["id"] in target_ids:
                records.append(record)
                if len(records) == 10:
                    break

    assert len(records) == 10, f"Expected 10 slice records, found {len(records)}"

    # Build retrieval contexts using BM25
    contexts: list[tuple[QueryRequest, RetrievalResult]] = []
    for r in records:
        paragraphs = r["paragraphs"]
        corpus = [tokenize(f"{p['title']} {p['paragraph_text']}") for p in paragraphs]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(tokenize(r["question"]))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]

        evidence = [
            EvidenceChunk(
                chunk_id=f"chunk-{paragraphs[i]['idx']}",
                text=f"{paragraphs[i]['title']}: {paragraphs[i]['paragraph_text']}",
                document_id=f"doc-{r['id']}-{paragraphs[i]['idx']}",
                source_file="dataset/musique",
                source_line=1,
            )
            for i in top_indices
        ]
        req = QueryRequest(
            question=r["question"],
            trace_id=f"trace-{r['id']}",
            origin="musique",
            record_id=r["id"],
            source_file="dataset/musique",
            source_line=1,
        )
        ret = RetrievalResult(
            trace_id=f"trace-{r['id']}",
            question=r["question"],
            mode="bm25",
            evidence=evidence,
        )
        contexts.append((req, ret))

    # Run 1: Generation with experiment profile
    settings = OpenRouterSettings(
        api_key="gate-m4-key",
        base_url=f"{base_url}/v1",
        llm_model="openai/gpt-4o-mini",
        embedding_model="openai/text-embedding-3-small",
        embedding_dimensions=1536,
        llm_max_tokens=256,
        require_parameters=True,
        allow_fallbacks=False,
        profile="experiment",
        temperature=0.0,
        seed=20261018,
        provider_order=["openai"],
    )
    llm = build_answer_llm(settings)

    run1_responses = []
    for req, ret in contexts:
        res = generate_answer(req, ret, llm)
        assert res.status == "succeeded", f"Answer generation failed: {res.error}"
        assert res.answer is not None and len(res.answer.strip()) > 0
        assert len(res.citations) > 0
        run1_responses.append(res)

    assert len(run1_responses) == 10

    # Gateway inspection: 10 calls, pinned provider, temperature 0, seed, no reasoning_effort
    with urllib.request.urlopen(f"{base_url}/requests") as resp:
        req_history = json.loads(resp.read().decode())["requests"]
        assert len(req_history) == 10
        for item in req_history:
            body = item["body"]
            assert body["temperature"] == 0.0
            assert body["seed"] == 20261018
            assert body["provider"]["order"] == ["openai"]
            assert body["provider"]["allow_fallbacks"] is False
            assert body["provider"]["require_parameters"] is True
            assert "reasoning_effort" not in body

    # Cost is logged
    with urllib.request.urlopen(f"{base_url}/health") as resp:
        health1 = json.loads(resp.read().decode())
        assert abs(health1["spent_usd"] - 0.010) < 1e-6
        assert health1["new_calls_count"] == 10

    # Run 2: Replay mode
    req_replay_cfg = urllib.request.Request(
        f"{base_url}/config",
        data=json.dumps({"mode": "replay", "reset_spend": True}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req_replay_cfg) as resp:
        assert resp.status == 200

    run2_responses = []
    for req, ret in contexts:
        res = generate_answer(req, ret, llm)
        assert res.status == "succeeded", f"Replay answer failed: {res.error}"
        run2_responses.append(res)

    # Outputs on Run 2 must be 100% identical to Run 1
    for r1, r2 in zip(run1_responses, run2_responses):
        assert r1.answer == r2.answer
        assert r1.citations == r2.citations
        assert r1.status == r2.status

    # Replay cost must be exactly $0
    with urllib.request.urlopen(f"{base_url}/health") as resp:
        health2 = json.loads(resp.read().decode())
        assert health2["spent_usd"] == 0.0
        assert health2["new_calls_count"] == 0
