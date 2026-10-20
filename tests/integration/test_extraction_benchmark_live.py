"""Opt-in standalone structured-extraction smoke tests; never touch Neo4j."""

import os
from pathlib import Path

import httpx
import pytest
from dotenv import load_dotenv

from graphrag_pipeline.extraction_benchmark.contracts import validate_extraction_response
from graphrag_pipeline.extraction_benchmark.artifacts import write_json
from graphrag_pipeline.extraction_benchmark.providers import EFFORTS, JSONProvider, ModelSettings
from graphrag_pipeline.extraction_benchmark.records import load_paragraphs


@pytest.mark.parametrize("provider", ["ollama", "openrouter"])
def test_live_benchmark_extraction(provider):
    flag = f"RUN_EXTRACTION_BENCHMARK_{provider.upper()}_LIVE"
    if os.getenv(flag) != "1":
        pytest.skip(f"Requires explicit {flag}=1 (OpenRouter incurs charges)")
    load_dotenv()
    model = os.getenv(f"BENCHMARK_{provider.upper()}_MODEL")
    if not model:
        pytest.skip(f"Specify BENCHMARK_{provider.upper()}_MODEL explicitly")
    effort = os.getenv("BENCHMARK_REASONING_EFFORT") if provider == "openrouter" else None
    if provider == "openrouter" and effort not in EFFORTS:
        pytest.skip("Specify an enabled BENCHMARK_REASONING_EFFORT explicitly")
    settings = ModelSettings.from_env(f"{provider}:{model}", reasoning_effort=effort)
    if provider == "ollama":
        try:
            response = httpx.get(settings.base_url + "/api/tags", timeout=3)
            response.raise_for_status()
        except httpx.HTTPError as error:
            pytest.skip(f"Ollama is unavailable: {type(error).__name__}")
    request = load_paragraphs(Path("tests/fixtures/musique/valid_records.jsonl"), count=1)[0]
    with JSONProvider(settings) as client:
        client.check()
        generated = client.generate(request)
    directory = Path(os.getenv("BENCHMARK_LIVE_EVIDENCE_DIR", "runs/entity-extraction/live-smoke"))
    directory.mkdir(parents=True, exist_ok=True)
    evidence = {"settings": settings.public(), "request": request.model_dump(mode="json"),
                "raw_content": generated.content, "metadata": generated.metadata}
    try:
        result = validate_extraction_response(request, generated.content)
    except ValueError as error:
        write_json(directory / f"{provider}.json", {**evidence, "status": "failed", "error": str(error)})
        raise
    write_json(directory / f"{provider}.json", {**evidence, "status": "succeeded", "result": result.model_dump(mode="json")})
    assert result.mentions, "fixture contains explicit entities; an empty extraction is not a successful smoke test"
