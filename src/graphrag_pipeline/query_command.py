"""Minimal command for the Neo4j GraphRAG question-to-answer path.

Last modified: 2026-10-06

Purpose of this file:
    Compose configured MuSiQue/custom question selection, optional bounded
    preload, Neo4j GraphRAG retrieval, grounded answer generation, correctness
    verification, and JSON run artifacts.

Files that use this file:
    ``python -m graphrag_pipeline.query_command`` is the user-facing command.
    Unit tests call ``execute_queries`` with fake retrievers and LLMs.

Caveats:
    A live run requires Neo4j, a verified vector index, and the configured
    embedding/answer provider. The command never preloads unless ``--preload``
    is explicitly supplied.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Callable

from graphrag_pipeline.answer import build_answer_llm, generate_answer
from graphrag_pipeline.config import (
    Neo4jSettings,
    dataset_paths_from_env,
    model_settings_from_env,
)
from graphrag_pipeline.musique_verification import (
    verify_query_correctness,
    write_run_artifact,
)
from graphrag_pipeline.query import (
    QueryRequest,
    QueryResponse,
    custom_query,
    run_query_batch,
    select_queries,
)
from graphrag_pipeline.retrieval import RetrievalError, RetrievalResult, RetrievalSettings, build_retriever, retrieve
from graphrag_pipeline.retrieval import (
    RetrievalError,
    RetrievalResult,
    RetrievalSettings,
    build_graphrag,
    build_retriever,
    retrieve,
)
from graphrag_pipeline.retrieval_prerequisites import verify_retrieval_prerequisites
from graphrag_pipeline.storage.neo4j_connection import Neo4jConnection


def _load_raw_record(dataset_paths: list[Path], record_id: str) -> dict[str, Any]:
    """Load the selected raw JSONL record for evaluation-only verification."""
    for path in dataset_paths:
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                if not raw_line.strip():
                    continue
                payload = json.loads(raw_line)
                if isinstance(payload, dict) and payload.get("id") == record_id:
                    return payload
    raise ValueError(f"record ID not found for correctness verification: {record_id}")


def _empty_retrieval(
    request: QueryRequest,
    mode: str,
    backend: str = "vectorrag",
) -> RetrievalResult:
    """Create an empty retrieval result for a failed retrieval/answer stage."""
    return RetrievalResult(
        trace_id=request.trace_id,
        question=request.question,
        mode=mode,
        evidence=[],
        backend=backend if backend in {"graphrag", "vectorrag"} else "vectorrag",
    )


def _failed_response(request: QueryRequest, mode: str, error: Exception) -> QueryResponse:
    """Convert retrieval/provider errors into the shared failed response contract."""
    return QueryResponse(
        status="failed",
        trace_id=request.trace_id,
        question=request.question,
        origin=request.origin,
        record_id=request.record_id,
        source_file=request.source_file,
        source_line=request.source_line,
        preload=request.preload,
        retrieval_backend=mode if mode in {"graphrag", "vectorrag"} else "vectorrag",
        retrieval_mode=mode,
        error=str(error),
    )


async def execute_queries(
    requests: list[QueryRequest],
    *,
    retriever: Any,
    llm: Any,
    retrieval_settings: RetrievalSettings,
    artifact_path: Path | None = None,
    dataset_paths: list[Path] | None = None,
    connection: Any | None = None,
    preload_loader: Callable[..., Any] | None = None,
    show_progress: bool = False,
    configuration: dict[str, Any] | None = None,
) -> list[QueryResponse]:
    """Execute selected requests through the shared retrieval and answer path.

    Args:
        requests: Validated custom or MuSiQue requests.
        retriever: Neo4j GraphRAG retriever, injectable for deterministic tests.
        llm: GraphRAG-supported answer LLM, injectable for deterministic tests.
        retrieval_settings: Retrieval mode, top-K, and context bound.
        artifact_path: Optional JSON path; batch outputs use numbered siblings.
        dataset_paths: Configured files used to recover raw MuSiQue records.
        connection: Optional read-only Neo4j connection for graph verification.
        preload_loader: Optional injected async loader for tests.
        show_progress: Whether the existing preload path shows progress.

    Returns:
        One structured response per request, preserving input order.
    """
    configured_paths = dataset_paths or []
    retrievals: dict[int, RetrievalResult] = {}

    async def answer_one(request: QueryRequest) -> QueryResponse:
        retrieval: RetrievalResult
        try:
            retrieval = retrieve(request, retriever, retrieval_settings=retrieval_settings)
            response = generate_answer(request, retrieval, llm)
        except RetrievalError as error:
            retrieval = _empty_retrieval(
                request,
                retrieval_settings.mode,
                retrieval_settings.backend,
            )
            response = _failed_response(request, retrieval_settings.backend, error)
        except Exception as error:  # noqa: BLE001 - command boundary returns structured failure
            retrieval = _empty_retrieval(
                request,
                retrieval_settings.mode,
                retrieval_settings.backend,
            )
            response = _failed_response(request, retrieval_settings.backend, error)
        retrievals[id(request)] = retrieval

        raw_record = None
        if request.origin == "musique" and request.record_id and configured_paths:
            raw_record = _load_raw_record(configured_paths, request.record_id)
        return response

    responses = await run_query_batch(
        requests,
        answer_one,
        preload_loader=preload_loader,
        show_progress=show_progress,
    )
    if artifact_path is not None:
        for index, (request, response) in enumerate(zip(requests, responses), start=1):
            retrieval = retrievals.get(
                id(request),
                _empty_retrieval(
                    request,
                    retrieval_settings.mode,
                    retrieval_settings.backend,
                ),
            )
            raw_record = None
            if request.origin == "musique" and request.record_id and configured_paths:
                raw_record = _load_raw_record(configured_paths, request.record_id)
            correctness = verify_query_correctness(
                request,
                retrieval,
                response,
                raw_record,
                connection=connection,
            )
            current_path = artifact_path
            if len(requests) > 1:
                current_path = artifact_path.with_name(
                    f"{artifact_path.stem}-{index}{artifact_path.suffix or '.json'}"
                )
            write_run_artifact(
                current_path,
                request,
                retrieval,
                response,
                correctness,
                configuration={
                    **(configuration or {}),
                    "retrieval_mode": retrieval_settings.mode,
                    "top_k": retrieval_settings.top_k,
                    "max_context_chunks": retrieval_settings.max_context_chunks,
                    "preload": request.preload,
                },
                errors=[response.error] if response.error else [],
            )
    return responses


def build_requests(args: argparse.Namespace) -> tuple[list[QueryRequest], list[Path]]:
    """Build custom or configured MuSiQue requests from parsed CLI arguments."""
    if args.question and (args.record_id or args.dataset_path):
        raise ValueError("--question cannot be combined with --record-id or --dataset-path")
    if args.question:
        return [custom_query(args.question)], []
    paths = [Path(item) for item in args.dataset_path] if args.dataset_path else dataset_paths_from_env()
    requests = select_queries(
        paths,
        record_id=args.record_id,
        max_records=args.max_records,
        preload=args.preload,
    )
    return requests, paths


def _resolve_retrieval_mode(backend: str, mode: str | None) -> str:
    """Default an unset retrieval mode from the selected backend.

    An explicitly supplied mode is always honored, including a conflicting
    one (which ``RetrievalSettings`` then rejects explicitly).
    """
    if mode is not None:
        return mode
    return "vector_cypher" if backend == "graphrag" else "vector"


def run_cli(args: argparse.Namespace) -> list[QueryResponse]:
    """Construct live Neo4j/provider dependencies and execute the CLI request."""
    requests, dataset_paths = build_requests(args)
    requests = [
        request.model_copy(update={"retrieval_backend": args.retrieval_backend})
        for request in requests
    ]
    neo4j_settings = Neo4jSettings.from_env()
    model_settings = model_settings_from_env()
    retrieval_settings = RetrievalSettings(
        backend=args.retrieval_backend,
        mode=_resolve_retrieval_mode(args.retrieval_backend, args.retrieval_mode),
        top_k=args.top_k,
        max_context_chunks=args.max_context_chunks,
        max_graph_nodes=args.max_graph_nodes,
    )
    connection = Neo4jConnection(neo4j_settings)
    connection.verify_connectivity()
    if not args.skip_prerequisite_check:
        verify_retrieval_prerequisites(
            connection,
            neo4j_settings,
            model_settings=model_settings,
        )
    vector_retriever = build_retriever(
        connection.connect(),
        neo4j_settings.database,
        model_settings,
        retrieval_settings=retrieval_settings,
    )
    llm = build_answer_llm(model_settings)
    retriever = (
        build_graphrag(vector_retriever, llm)
        if args.retrieval_backend == "graphrag"
        else vector_retriever
    )
    try:
        return asyncio.run(
            execute_queries(
                requests,
                retriever=retriever,
                llm=llm,
                retrieval_settings=retrieval_settings,
                artifact_path=args.artifact,
                dataset_paths=dataset_paths,
                connection=connection,
                show_progress=args.progress,
                configuration={
                    "neo4j_database": neo4j_settings.database,
                    "provider": type(model_settings).__name__,
                    "llm_model": model_settings.llm_model,
                    "embedding_model": model_settings.embedding_model,
                    "embedding_dimensions": getattr(model_settings, "embedding_dimensions", None),
                    "retrieval_backend": retrieval_settings.backend,
                },
            )
        )
    finally:
        connection.close()


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Run one bounded Neo4j GraphRAG QA query batch.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--question", help="Custom question; no MuSiQue provenance is assigned.")
    source.add_argument("--record-id", help="Exact MuSiQue record ID from configured JSONL files.")
    parser.add_argument("--dataset-path", action="append", help="JSONL source path; repeat for multiple files.")
    parser.add_argument("--max-records", type=int, help="Bound the selected MuSiQue batch.")
    parser.add_argument("--preload", action="store_true", help="Preload selected MuSiQue records before QA.")
    parser.add_argument("--retrieval-backend", choices=("graphrag", "vectorrag"), default="vectorrag")
    parser.add_argument(
        "--retrieval-mode",
        choices=("vector", "vector_cypher"),
        default=None,
        help="Defaults to vector_cypher for --retrieval-backend graphrag, vector otherwise.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-context-chunks", type=int, default=5)
    parser.add_argument(
        "--max-graph-nodes",
        type=int,
        default=200,
        help="Resource budget for one seed chunk's entity-subgraph traversal (graphrag/vector_cypher only).",
    )
    parser.add_argument("--artifact", type=Path, help="Write one JSON artifact per query (numbered for batches).")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--skip-prerequisite-check", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    """Run the command and print concise JSON responses."""
    args = _parser().parse_args()
    try:
        responses = run_cli(args)
    except Exception as error:  # noqa: BLE001 - CLI reports a nonzero structured failure
        print(json.dumps({"status": "failed", "error": str(error)}, sort_keys=True))
        return 1
    print(json.dumps([response.model_dump(mode="json") for response in responses], indent=2, sort_keys=True))
    return 0 if all(response.status != "failed" for response in responses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
