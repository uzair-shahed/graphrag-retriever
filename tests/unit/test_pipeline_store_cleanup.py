"""Tests verifying SimpleKGPipeline result store cleanup and flat memory growth.

Step M3.5 in the implementation plan:
- Verifies that pipeline.runner.pipeline.store and final_results exist.
- Verifies that clearing them after each run keeps store entries at 0.
- Fails loudly if a library upgrade modifies or removes the internal store paths.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from neo4j import GraphDatabase

from graphrag_pipeline.config import OpenRouterSettings
from graphrag_pipeline.extraction.pipeline import (
    build_simple_kg_pipeline,
    clear_pipeline_store,
)


@pytest.fixture
def dummy_settings() -> OpenRouterSettings:
    return OpenRouterSettings(
        api_key="sk-test",
        base_url="http://localhost:18080/v1",
        llm_model="openai/gpt-4o-mini",
        embedding_model="openai/text-embedding-3-small",
        embedding_dimensions=1536,
        llm_max_tokens=100,
        require_parameters=True,
        allow_fallbacks=True,
    )


@pytest.fixture
def dummy_pipeline(dummy_settings: OpenRouterSettings):
    driver = GraphDatabase.driver("bolt://127.0.0.1:7687", auth=("test", "test"))
    try:
        with patch.object(
            driver,
            "execute_query",
            MagicMock(
                return_value=(
                    [{"versions": ["2026.08.1"], "edition": "enterprise"}],
                    None,
                    None,
                )
            ),
        ):
            yield build_simple_kg_pipeline(driver, dummy_settings)
    finally:
        driver.close()


def test_pipeline_store_internal_paths_exist(dummy_pipeline) -> None:
    """SimpleKGPipeline must expose runner.pipeline.store and final_results with empty().

    If a neo4j-graphrag library upgrade moves or removes these attributes, this test
    fails loudly to alert developers to adjust the store clearing paths.
    """
    pipeline = dummy_pipeline

    assert hasattr(pipeline, "runner"), "SimpleKGPipeline must have 'runner'"
    runner = pipeline.runner
    assert hasattr(runner, "pipeline"), "PipelineRunner must have 'pipeline'"
    inner = runner.pipeline

    assert hasattr(inner, "store"), "Inner pipeline must have 'store'"
    assert hasattr(inner.store, "empty"), "store must have 'empty()' method"
    assert hasattr(inner.store, "all"), "store must have 'all()' method"

    assert hasattr(inner, "final_results"), "Inner pipeline must have 'final_results'"
    assert hasattr(inner.final_results, "empty"), "final_results must have 'empty()' method"
    assert hasattr(inner.final_results, "all"), "final_results must have 'all()' method"


def test_clear_pipeline_store_empties_both_stores(dummy_pipeline) -> None:
    """clear_pipeline_store must empty both store and final_results."""
    pipeline = dummy_pipeline
    store = pipeline.runner.pipeline.store
    final_results = pipeline.runner.pipeline.final_results

    # Simulate entries in store and final_results
    store._data["comp1"] = {"status": "SUCCESS", "data": "output1"}
    store._data["comp2"] = {"status": "SUCCESS", "data": "output2"}
    final_results._data["run_1"] = {"result": "ok"}

    assert len(store.all()) == 2
    assert len(final_results.all()) == 1

    clear_pipeline_store(pipeline)

    assert len(store.all()) == 0
    assert len(final_results.all()) == 0


def test_clear_pipeline_store_handles_none_gracefully() -> None:
    """clear_pipeline_store must not raise if pipeline or runner is None."""
    clear_pipeline_store(None)
    clear_pipeline_store(MagicMock())


def test_store_stays_empty_across_repeated_runs(dummy_pipeline) -> None:
    """Simulating 200 runs on one pipeline instance maintains 0 store entries when cleared."""
    pipeline = dummy_pipeline
    store = pipeline.runner.pipeline.store
    final_results = pipeline.runner.pipeline.final_results

    for i in range(200):
        # Simulate intermediate data generated during a run (typically ~12 entries)
        for c in range(12):
            store._data[f"comp_{c}_run_{i}"] = f"data_{c}"
        final_results._data[f"run_{i}"] = "result"

        # After run: clear store
        clear_pipeline_store(pipeline)

        # Invariant: store entries stay 0
        assert len(store.all()) == 0
        assert len(final_results.all()) == 0

