"""Preflight checks running before ingestion or retrieval to fail fast for $0."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from neo4j.exceptions import ServiceUnavailable

from graphrag_pipeline.config import (
    Neo4jSettings,
    OllamaSettings,
    OpenRouterSettings,
)
from graphrag_pipeline.storage.neo4j_connection import Neo4jConnection


class PreflightError(RuntimeError):
    """Raised when environment or database prerequisites fail before spend."""


def check_neo4j_connectivity_and_apoc(settings: Neo4jSettings) -> None:
    """Verify Neo4j is reachable and required APOC procedures exist."""
    try:
        with Neo4jConnection(settings) as conn:
            conn.verify_connectivity()
            # Check required APOC procedures used by SimpleKGPipeline writer
            procs = conn.run_read(
                "SHOW PROCEDURES YIELD name WHERE name IN ['apoc.create.addLabels', 'apoc.merge.relationship'] RETURN name"
            )
            found = {p["name"] for p in procs}
            missing = {"apoc.create.addLabels", "apoc.merge.relationship"} - found
            if missing:
                raise PreflightError(
                    f"Missing required APOC procedures: {sorted(missing)}. "
                    "Ensure APOC plugin is enabled in Neo4j."
                )
    except PreflightError:
        raise
    except ServiceUnavailable as error:
        raise PreflightError(
            f"Neo4j is not reachable at {settings.uri}: {error}"
        ) from error
    except Exception as error:
        raise PreflightError(f"Neo4j connectivity check failed: {error}") from error


def check_vector_index(settings: Neo4jSettings, expected_dim: int | None = 1536) -> None:
    """Verify chunk_embedding vector index exists, is ONLINE, and dimensions match embedder."""
    if expected_dim is None:
        return
    try:
        with Neo4jConnection(settings) as conn:
            indexes = conn.run_read(
                "SHOW VECTOR INDEXES YIELD name, labelsOrTypes, properties, dimensions "
                "WHERE name = 'chunk_embedding' RETURN dimensions, labelsOrTypes[0] AS label, properties[0] AS property"
            )
            if not indexes:
                raise PreflightError("Vector index 'chunk_embedding' does not exist in Neo4j.")
            actual_dim = indexes[0].get("dimensions")
            if actual_dim != expected_dim:
                raise PreflightError(
                    f"Vector index dimension mismatch: index has {actual_dim}, but embedder requires {expected_dim}."
                )

            # Also verify index state is ONLINE
            state_rows = conn.run_read(
                "SHOW INDEXES YIELD name, state WHERE name = 'chunk_embedding' RETURN state"
            )
            if state_rows:
                state = state_rows[0].get("state")
                if state is not None and state != "ONLINE":
                    raise PreflightError(
                        f"Vector index 'chunk_embedding' is not ONLINE: state is {state}."
                    )
    except PreflightError:
        raise
    except Exception as error:
        raise PreflightError(f"Vector index check failed: {error}") from error


def check_budget_configured(model_settings: OpenRouterSettings | OllamaSettings) -> None:
    """Verify that a budget ceiling is configured before paid calls can be made."""
    if isinstance(model_settings, OpenRouterSettings):
        budget_str = os.environ.get("OPENROUTER_BUDGET_USD") or os.environ.get("GRAPH_RAG_BUDGET_USD")
        if budget_str:
            try:
                val = float(budget_str)
                if val <= 0:
                    raise PreflightError(f"Budget ceiling must be positive, got {val}")
                return
            except ValueError as error:
                raise PreflightError(f"Invalid budget setting: {budget_str}") from error

        # Check if model_settings.base_url is gateway with /health endpoint reporting budget
        health_url = model_settings.base_url.rstrip("/")
        if health_url.endswith("/v1"):
            health_url = health_url[:-3]
        health_url += "/health"
        try:
            req = urllib.request.Request(health_url)
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode())
                    if data.get("budget_usd") is not None and float(data["budget_usd"]) > 0:
                        return
        except Exception:
            pass

        raise PreflightError(
            "A budget ceiling must be configured before making paid calls "
            "(set OPENROUTER_BUDGET_USD or pass --budget-usd to model gateway)."
        )


def check_model_endpoint(model_settings: OpenRouterSettings | OllamaSettings) -> None:
    """Verify model provider endpoint is reachable."""
    if isinstance(model_settings, OpenRouterSettings):
        url = model_settings.base_url.rstrip("/") + "/models"
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {model_settings.api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status not in (200, 204):
                    raise PreflightError(f"Model endpoint returned HTTP {resp.status}")
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise PreflightError(f"Model endpoint authentication failed: HTTP {e.code}") from e
            raise PreflightError(f"Model endpoint returned HTTP {e.code}") from e
        except Exception as e:
            raise PreflightError(f"Model endpoint is unreachable at {url}: {e}") from e
    elif isinstance(model_settings, OllamaSettings):
        url = model_settings.host.rstrip("/") + "/api/tags"
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status != 200:
                    raise PreflightError(f"Ollama endpoint returned HTTP {resp.status}")
        except Exception as e:
            raise PreflightError(f"Ollama endpoint is unreachable at {url}: {e}") from e


def run_preflight_checks(
    neo4j_settings: Neo4jSettings,
    model_settings: OpenRouterSettings | OllamaSettings | None = None,
    require_vector_index: bool = False,
) -> None:
    """Run all preflight checks; fails in seconds for $0 before any paid work."""
    check_neo4j_connectivity_and_apoc(neo4j_settings)
    if model_settings is not None:
        check_budget_configured(model_settings)
        check_model_endpoint(model_settings)
    if require_vector_index and model_settings is not None:
        dim = getattr(model_settings, "embedding_dimensions", 1536)
        check_vector_index(neo4j_settings, expected_dim=dim)
