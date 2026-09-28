"""MuSiQue dataset ingestion: Contract 1 record-to-chunk adaptation."""

from graphrag_pipeline.ingestion.record_adapter import (
    RecordAdapterError,
    build_requests_for_record,
    compute_file_sha256,
    iter_jsonl_lines,
    slugify_title,
)

__all__ = [
    "RecordAdapterError",
    "build_requests_for_record",
    "compute_file_sha256",
    "iter_jsonl_lines",
    "slugify_title",
]
