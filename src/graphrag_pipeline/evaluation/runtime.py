"""Read-only graph checks and controlled retrieval for the paired benchmark."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from graphrag_pipeline.answer_context import render_evidence_context
from graphrag_pipeline.ingestion.reconcile import COMPLETE, classify_chunk
from graphrag_pipeline.retrieval import (
    RetrievalSettings, _evidence_from_item, _unique_items, build_retriever,
)


@dataclass(frozen=True)
class EvaluationSettings:
    top_k: int = 5
    max_context_chunks: int = 5
    max_context_tokens: int = 4096
    max_graph_nodes: int = 200
    conditions: tuple[str, ...] = ("record", "corpus")

    def __post_init__(self):
        if min(self.top_k, self.max_context_chunks, self.max_context_tokens, self.max_graph_nodes) < 1:
            raise ValueError("retrieval and context limits must be positive")
        if not self.conditions or len(set(self.conditions)) != len(self.conditions) or set(self.conditions) - {"record", "corpus"}:
            raise ValueError("conditions must be unique record/corpus values")


def graph_readiness(connection, cases):
    issues = []
    for case in cases:
        for request in case.extraction_requests():
            state = classify_chunk(connection, request)
            if state.state != COMPLETE:
                issues.append(state.to_report(request))
    return {"ready": not issues, "selected_records": len(cases),
            "missing_record_ids": sorted({i["record_id"] for i in issues if i["state"] == "absent"}),
            "issues": issues}


def graph_snapshot(connection, database):
    """Fingerprint all data and index definitions using streamed read-only queries.

    Use a quiet database. Before/after hashes detect persistent concurrent
    changes; they cannot supply snapshot isolation across independent searches.
    """
    digest = hashlib.sha256()
    counts = {}
    with connection.connect().session(database=database, default_access_mode="READ") as session:
        for name, query in (
            ("nodes", "MATCH (n) RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS properties ORDER BY id"),
            ("relationships", "MATCH (a)-[r]->(b) RETURN elementId(r) AS id, elementId(a) AS start, elementId(b) AS end, type(r) AS type, properties(r) AS properties ORDER BY id"),
            ("indexes", "SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, properties, options, state RETURN * ORDER BY name"),
        ):
            count = 0
            digest.update(name.encode())
            for row in session.run(query):
                data = row.data()
                if "labels" in data:
                    data["labels"] = sorted(data["labels"])
                digest.update(json.dumps(data, sort_keys=True, ensure_ascii=False, default=str).encode())
                digest.update(b"\n")
                count += 1
            counts[name] = count
    return {"sha256": digest.hexdigest(), **counts}


class PairedRetriever:
    """Same embedded question and vector seed count; only graph expansion differs."""

    def __init__(self, driver, database, model_settings, settings, index_settings=None):
        self.settings = settings
        self.retrievers = {}
        for backend, mode in (("vectorrag", "vector"), ("graphrag", "vector_cypher")):
            self.retrievers[backend] = build_retriever(
                driver, database, model_settings, index_settings=index_settings,
                retrieval_settings=RetrievalSettings(backend=backend, mode=mode, top_k=settings.top_k,
                    max_context_chunks=settings.max_context_chunks, max_graph_nodes=settings.max_graph_nodes),
            )

    def embed(self, question):
        return self.retrievers["vectorrag"].embedder.embed_query(question)

    def retrieve(self, backend, vector, record_id):
        kwargs = {"query_vector": vector, "top_k": self.settings.top_k}
        if record_id is not None:
            kwargs["filters"] = {"record_id": record_id}
        if backend == "graphrag":
            kwargs["query_params"] = {"node_limit": self.settings.max_graph_nodes, "query_limit": self.settings.max_graph_nodes + 1}
        raw = self.retrievers[backend].search(**kwargs)
        return [_evidence_from_item(i) for i in _unique_items(raw.items)]


def select_context(evidence, settings, count_tokens):
    """Select a ranked prefix of whole evidence blocks. Never cut graph lines/text."""
    selected = []
    for item in evidence[:settings.max_context_chunks]:
        candidate = [*selected, item]
        if count_tokens(render_evidence_context(candidate)) > settings.max_context_tokens:
            break
        selected = candidate
    return selected
