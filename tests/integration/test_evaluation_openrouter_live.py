"""Paid OpenRouter smoke test, written but deliberately not run for this task."""

import asyncio
import json
import os
from pathlib import Path

import pytest

from graphrag_pipeline.answer import build_answer_llm, generate_answer
from graphrag_pipeline.config import OpenRouterSettings
from graphrag_pipeline.evaluation.providers import EvaluatorSettings, build_metrics
from graphrag_pipeline.evaluation.scoring import normalize_answer, triad_scores
from graphrag_pipeline.query import EvidenceChunk, custom_query
from graphrag_pipeline.retrieval import RetrievalResult


def test_openrouter_answer_and_ragas_triad():
    if os.getenv("RUN_OPENROUTER_EVALUATION_LIVE") != "1":
        pytest.skip("Paid OpenRouter test requires explicit RUN_OPENROUTER_EVALUATION_LIVE=1.")
    settings = OpenRouterSettings.from_env()
    request = custom_query("Where is Acme headquartered?")
    evidence = [EvidenceChunk(chunk_id="openrouter-smoke-1", text="Acme is headquartered in Boston.")]
    response = generate_answer(request, RetrievalResult(request.trace_id, request.question, "vector", evidence), build_answer_llm(settings))
    assert response.status == "succeeded" and response.citations == [evidence[0].chunk_id]
    assert normalize_answer(response.answer) in {"boston", "boston massachusetts"}
    evaluator = EvaluatorSettings("openrouter", settings.llm_model, settings.embedding_model,
                                  settings.base_url, settings.api_key)
    async def score():
        metrics, client = build_metrics(evaluator)
        try:
            return await triad_scores(metrics, response, evidence, timeout=evaluator.timeout)
        finally:
            await client.close()
    scores = asyncio.run(score())
    directory = Path(os.getenv("OPENROUTER_EVALUATION_EVIDENCE_DIR", "runs/openrouter-evaluation-live"))
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "result.json").write_text(json.dumps({"response": response.model_dump(mode="json"), "ragas": scores}, indent=2))
    assert all(metric["status"] == "scored" for metric in scores.values())
