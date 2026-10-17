"""Load a bounded slice of a MuSiQue JSONL dataset through SimpleKGPipeline.

This script is the batch counterpart to ``run_fixture.py``: instead of one
hand-written fixture paragraph, it reads a contiguous slice of records from an
official MuSiQue JSONL file (paragraph-as-chunk, per
``agent-coordination/data-contracts.md`` Contract 1) and sends every paragraph
of every selected record through the live Neo4j GenAI Plugin pipeline.

Each MuSiQue record contains many paragraphs (the full dev split averages
about 20 per record), so requesting even a small number of records can send
dozens of paragraphs through the pipeline. Use ``--dry-run`` first to see the
exact chunk count and identities before spending any OpenRouter budget.

Example:
    uv run --locked python -m graphrag_pipeline.ingestion.load_musique_dataset \\
        --records 3 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tqdm import tqdm

from graphrag_pipeline.config import Neo4jSettings, model_settings_from_env
from graphrag_pipeline.diagnostics import (
    exception_diagnostics,
    save_invalid_llm_response,
)
from graphrag_pipeline.extraction.pipeline import build_simple_kg_pipeline
from graphrag_pipeline.extraction.plugin_input import build_plugin_input
from graphrag_pipeline.extraction.schemas import ExtractionRequest
from graphrag_pipeline.ingestion.record_adapter import (
    RecordAdapterError,
    build_requests_for_record,
    compute_file_sha256,
    iter_jsonl_lines,
)
from graphrag_pipeline.storage.neo4j_connection import Neo4jConnection

if TYPE_CHECKING:
    from graphrag_pipeline.query import QueryRequest

DEFAULT_DATASET_PATH = Path("dataset/musique/data/data/musique_full_v1.0_dev.jsonl")


@dataclass
class LoadSummary:
    """Outcome of one batch load run."""

    dataset: str
    source_split: str
    start_record: int
    records_requested: int
    records_read: int
    dead_lettered_records: list[dict[str, Any]] = field(default_factory=list)
    chunks_planned: int = 0
    records_planned: int = 0
    chunks_succeeded: list[dict[str, Any]] = field(default_factory=list)
    chunks_failed: list[dict[str, Any]] = field(default_factory=list)
    records_succeeded: int = 0
    collect_seconds: float = 0.0
    run_seconds: float = 0.0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "source_split": self.source_split,
            "start_record": self.start_record,
            "records_requested": self.records_requested,
            "records_read": self.records_read,
            "records_planned": self.records_planned,
            "records_loaded": self.records_succeeded,
            "chunks_planned": self.chunks_planned,
            "chunks_succeeded": len(self.chunks_succeeded),
            "chunks_failed": len(self.chunks_failed),
            "dead_lettered_records": self.dead_lettered_records,
            "failed_chunks": self.chunks_failed,
            "collect_seconds": round(self.collect_seconds, 3),
            "run_seconds": round(self.run_seconds, 3),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "elapsed": format_duration(self.elapsed_seconds),
        }


@dataclass
class PreloadSummary:
    """Outcome of preloading the records selected for a query run."""

    requested_records: list[dict[str, Any]]
    records_read: int = 0
    dead_lettered_records: list[dict[str, Any]] = field(default_factory=list)
    chunks_planned: int = 0
    chunks_succeeded: list[dict[str, Any]] = field(default_factory=list)
    chunks_failed: list[dict[str, Any]] = field(default_factory=list)
    records_succeeded: int = 0
    collect_seconds: float = 0.0
    run_seconds: float = 0.0
    elapsed_seconds: float = 0.0

    @property
    def succeeded(self) -> bool:
        """Return whether every selected record was loaded successfully."""
        return not self.dead_lettered_records and not self.chunks_failed

    def to_dict(self) -> dict[str, Any]:
        """Return an audit-friendly representation of the preload outcome."""
        return {
            "requested_records": self.requested_records,
            "records_read": self.records_read,
            "records_loaded": self.records_succeeded,
            "chunks_planned": self.chunks_planned,
            "chunks_succeeded": len(self.chunks_succeeded),
            "chunks_failed": self.chunks_failed,
            "dead_lettered_records": self.dead_lettered_records,
            "collect_seconds": round(self.collect_seconds, 3),
            "run_seconds": round(self.run_seconds, 3),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "elapsed": format_duration(self.elapsed_seconds),
        }


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as a compact, human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, remainder_seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remainder_seconds:.1f}s"
    hours, remainder_minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(remainder_minutes)}m {remainder_seconds:.1f}s"


def measure_duration_seconds(start: float) -> float:
    """Return the elapsed seconds since a ``time.perf_counter()`` start value."""
    return time.perf_counter() - start


def synchronize_chunk_provenance(connection: Neo4jConnection, chunk_id: str) -> None:
    """Copy application provenance from a plugin Document onto its Chunk nodes.

    Args:
        connection: Neo4j connection used for the bounded write.
        chunk_id: Application chunk ID stored on the plugin-created Document.

    Effects:
        Makes chunk-level retrieval results self-describing for the vector
        retriever and read-only provenance verifier.

    Raises:
        RuntimeError: If the plugin-created document cannot be found.
    """
    rows = connection.run_write(
        """
        MATCH (c:Chunk)-[:FROM_DOCUMENT]->(d:Document)
        WHERE d.chunk_id = $chunk_id
        SET c.chunk_id = d.chunk_id,
            c.record_id = d.record_id,
            c.document_id = d.document_id,
            c.source_file = d.source_file,
            c.source_line = d.source_line,
            c.source_sha256 = d.source_sha256,
            c.paragraph_idx = d.paragraph_idx,
            c.title = d.title
        RETURN count(c) AS chunks
        """,
        {"chunk_id": chunk_id},
    )
    if not rows or rows[0].get("chunks", 0) < 1:
        raise RuntimeError(f"plugin-created document not found for chunk {chunk_id!r}")


def derive_source_split(dataset_path: Path) -> str:
    """Infer a provenance split label such as ``full_dev`` from the file name."""
    parts = dataset_path.stem.split("_")
    # e.g. "musique_full_v1.0_dev" -> ["musique", "full", "v1.0", "dev"]
    if len(parts) >= 4 and parts[0] == "musique":
        return f"{parts[1]}_{parts[-1]}"
    return dataset_path.stem


def collect_requests(
    dataset_path: Path,
    start_record: int,
    max_records: int,
    source_split: str,
) -> tuple[list[ExtractionRequest], list[dict[str, Any]], int]:
    """Read a bounded slice of records and adapt them into chunk requests.

    Returns the validated requests, dead-lettered record summaries, and the
    number of records actually read from the file (which can be less than
    ``max_records`` if the file ends early).
    """
    source_sha256 = compute_file_sha256(dataset_path)
    source_file = str(dataset_path)

    requests: list[ExtractionRequest] = []
    dead_letters: list[dict[str, Any]] = []
    records_read = 0

    for source_line, raw_line in iter_jsonl_lines(dataset_path, start_record, max_records):
        records_read += 1
        try:
            record_requests = build_requests_for_record(
                raw_line=raw_line,
                source_line=source_line,
                source_file=source_file,
                source_sha256=source_sha256,
                source_split=source_split,
            )
        except RecordAdapterError as error:
            dead_letters.append({"source_line": source_line, "reason": error.reason})
            continue
        requests.extend(record_requests)

    return requests, dead_letters, records_read


def collect_selected_requests(
    selected_requests: list[QueryRequest],
) -> tuple[list[ExtractionRequest], list[dict[str, Any]], int]:
    """Adapt exactly the JSONL records represented by validated query requests.

    Args:
        selected_requests: MuSiQue requests retaining source file and line.

    Returns:
        Chunk requests, dead-letter summaries, and the number of records read.

    Raises:
        ValueError: If a MuSiQue request does not include source provenance.
    """
    requests: list[ExtractionRequest] = []
    dead_letters: list[dict[str, Any]] = []
    checksums: dict[Path, str] = {}

    for selected in selected_requests:
        if selected.origin != "musique" or not selected.source_file or selected.source_line is None:
            raise ValueError("preload requires MuSiQue source_file and source_line provenance")

        path = Path(selected.source_file)
        if path not in checksums:
            checksums[path] = compute_file_sha256(path)

        raw_line: str | None = None
        with path.open("r", encoding="utf-8") as handle:
            for line_number, candidate in enumerate(handle, start=1):
                if line_number == selected.source_line:
                    raw_line = candidate.strip()
                    break

        if raw_line is None or not raw_line:
            dead_letters.append(
                {
                    "record_id": selected.record_id,
                    "source_line": selected.source_line,
                    "reason": "selected source line is unavailable",
                }
            )
            continue

        try:
            record_requests = build_requests_for_record(
                raw_line=raw_line,
                source_line=selected.source_line,
                source_file=str(path),
                source_sha256=checksums[path],
                source_split=selected.source_split or derive_source_split(path),
            )
        except RecordAdapterError as error:
            dead_letters.append(
                {
                    "record_id": selected.record_id,
                    "source_line": selected.source_line,
                    "reason": error.reason,
                }
            )
            continue

        if not record_requests or record_requests[0].record_id != selected.record_id:
            dead_letters.append(
                {
                    "record_id": selected.record_id,
                    "source_line": selected.source_line,
                    "reason": "selected source record ID does not match the query request",
                }
            )
            continue
        requests.extend(record_requests)

    return requests, dead_letters, len(selected_requests)


async def run_requests(
    requests: list[ExtractionRequest],
    neo4j_settings: Neo4jSettings,
    openrouter_settings: object,
    show_progress: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Send every validated chunk request through the live pipeline in order.

    One pipeline instance is reused for the batch so schema setup is not
    repeated for every paragraph. A failure on one chunk is recorded and does
    not stop the remaining chunks.

    A tqdm progress bar tracks chunk-level progress and reports, as a running
    postfix, how many records have already had every one of their chunks
    succeed. This assumes ``requests`` groups each record's chunks together,
    which is how ``collect_requests`` builds the list.
    """
    succeeded: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    total_records = len({request.record_id for request in requests})
    records_loaded = 0
    current_record_id: str | None = None
    current_record_failed = False

    connection = Neo4jConnection(neo4j_settings)
    try:
        driver = connection.connect()
        pipeline = build_simple_kg_pipeline(driver, openrouter_settings)
        progress = tqdm(
            total=len(requests),
            desc="Loading MuSiQue chunks",
            unit="chunk",
            disable=not show_progress,
        )
        try:
            for request in requests:
                if request.record_id != current_record_id:
                    if current_record_id is not None and not current_record_failed:
                        records_loaded += 1
                    current_record_id = request.record_id
                    current_record_failed = False

                text, document_metadata = build_plugin_input(request)
                try:
                    result = await pipeline.run_async(text=text, document_metadata=document_metadata)
                    synchronize_chunk_provenance(connection, request.chunk_id)
                except Exception as error:  # noqa: BLE001 - recorded, not swallowed
                    current_record_failed = True
                    failure = {
                        "chunk_id": request.chunk_id,
                        "record_id": request.record_id,
                        **exception_diagnostics(error),
                    }
                    if pipeline is not None:
                        llm = pipeline.runner.config.llm_config["default"].root
                        response_path = save_invalid_llm_response(
                            error,
                            getattr(llm, "last_response", None),
                        )
                        if response_path is not None:
                            failure["invalid_llm_response"] = response_path.as_posix()
                    failed.append(failure)
                else:
                    succeeded.append(
                        {
                            "chunk_id": request.chunk_id,
                            "record_id": request.record_id,
                            "run_id": result.run_id,
                        }
                    )
                progress.update(1)
                progress.set_postfix(records_loaded=f"{records_loaded}/{total_records}", failed=len(failed))

            if current_record_id is not None and not current_record_failed:
                records_loaded += 1
                progress.set_postfix(records_loaded=f"{records_loaded}/{total_records}", failed=len(failed))
        finally:
            progress.close()
    finally:
        connection.close()

    return succeeded, failed


def count_fully_succeeded_records(
    requests: list[ExtractionRequest], failed: list[dict[str, Any]]
) -> int:
    """Count distinct records for which every planned chunk succeeded."""
    attempted_record_ids = {request.record_id for request in requests}
    failed_record_ids = {item["record_id"] for item in failed}
    return len(attempted_record_ids - failed_record_ids)


async def load_dataset(
    dataset_path: Path,
    start_record: int,
    max_records: int,
    source_split: str | None,
    dry_run: bool,
    show_progress: bool = True,
) -> LoadSummary:
    """Adapt, and unless ``dry_run``, run a bounded slice of a MuSiQue file."""
    started_at = time.perf_counter()
    resolved_split = source_split or derive_source_split(dataset_path)

    collect_started_at = time.perf_counter()
    requests, dead_letters, records_read = collect_requests(
        dataset_path, start_record, max_records, resolved_split
    )
    collect_seconds = measure_duration_seconds(collect_started_at)

    summary = LoadSummary(
        dataset=str(dataset_path),
        source_split=resolved_split,
        start_record=start_record,
        records_requested=max_records,
        records_read=records_read,
        dead_lettered_records=dead_letters,
        chunks_planned=len(requests),
        records_planned=len({request.record_id for request in requests}),
        collect_seconds=collect_seconds,
    )

    if dry_run:
        summary.elapsed_seconds = measure_duration_seconds(started_at)
        return summary

    neo4j_settings = Neo4jSettings.from_env()
    model_settings = model_settings_from_env()

    run_started_at = time.perf_counter()
    succeeded, failed = await run_requests(
        requests, neo4j_settings, model_settings, show_progress=show_progress
    )
    summary.run_seconds = measure_duration_seconds(run_started_at)

    summary.chunks_succeeded = succeeded
    summary.chunks_failed = failed
    summary.records_succeeded = count_fully_succeeded_records(requests, failed)
    summary.elapsed_seconds = measure_duration_seconds(started_at)
    return summary


async def load_selected_records(
    selected_requests: list[QueryRequest],
    *,
    dry_run: bool = False,
    show_progress: bool = True,
    neo4j_settings: Neo4jSettings | None = None,
    openrouter_settings: object | None = None,
) -> PreloadSummary:
    """Preload selected MuSiQue records through the existing ingestion path.

    Args:
        selected_requests: Validated MuSiQue query requests to preload.
        dry_run: Plan and validate chunks without connecting to providers.
        show_progress: Whether live loading reports chunk progress.
        neo4j_settings: Optional injected Neo4j settings for tests or callers.
        openrouter_settings: Optional injected model settings for tests or callers.

    Returns:
        A summary containing selected-record metadata, timings, dead letters,
        and chunk-level loader outcomes.
    """
    started_at = time.perf_counter()
    requested_records = [
        {
            "record_id": request.record_id,
            "source_file": request.source_file,
            "source_line": request.source_line,
            "trace_id": request.trace_id,
        }
        for request in selected_requests
    ]
    collect_started_at = time.perf_counter()
    requests, dead_letters, records_read = collect_selected_requests(selected_requests)
    collect_seconds = measure_duration_seconds(collect_started_at)
    summary = PreloadSummary(
        requested_records=requested_records,
        records_read=records_read,
        dead_lettered_records=dead_letters,
        chunks_planned=len(requests),
        collect_seconds=collect_seconds,
    )

    if dry_run or not requests:
        summary.elapsed_seconds = measure_duration_seconds(started_at)
        return summary

    resolved_neo4j_settings = neo4j_settings or Neo4jSettings.from_env()
    resolved_model_settings = openrouter_settings or model_settings_from_env()
    run_started_at = time.perf_counter()
    succeeded, failed = await run_requests(
        requests,
        resolved_neo4j_settings,
        resolved_model_settings,
        show_progress=show_progress,
    )
    summary.run_seconds = measure_duration_seconds(run_started_at)
    summary.chunks_succeeded = succeeded
    summary.chunks_failed = failed
    summary.records_succeeded = count_fully_succeeded_records(requests, failed)
    summary.elapsed_seconds = measure_duration_seconds(started_at)
    return summary


def main() -> None:
    """Parse CLI arguments, run the batch load, and print a JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to an official MuSiQue JSONL file.",
    )
    parser.add_argument(
        "--records",
        type=int,
        required=True,
        help="Number of MuSiQue records to add, starting at --start-record.",
    )
    parser.add_argument(
        "--start-record",
        type=int,
        default=1,
        help="1-indexed record number to start from (default: 1).",
    )
    parser.add_argument(
        "--source-split",
        type=str,
        default=None,
        help="Provenance split label. Derived from the file name if omitted.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and count chunks without calling Neo4j or OpenRouter.",
    )
    parser.add_argument(
        "--no-progress",
        dest="show_progress",
        action="store_false",
        help="Disable the tqdm progress bar during a live run.",
    )
    parser.set_defaults(show_progress=True)
    args = parser.parse_args()

    summary = asyncio.run(
        load_dataset(
            dataset_path=args.dataset_path,
            start_record=args.start_record,
            max_records=args.records,
            source_split=args.source_split,
            dry_run=args.dry_run,
            show_progress=args.show_progress,
        )
    )
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    if summary.chunks_failed or summary.dead_lettered_records:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
