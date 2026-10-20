"""Select a reproducible paragraph prefix without loading evaluation labels."""

from pathlib import Path

from graphrag_pipeline.extraction.schemas import ExtractionRequest
from graphrag_pipeline.ingestion.record_adapter import (
    build_requests_for_record,
    compute_file_sha256,
    iter_jsonl_lines,
)

DEFAULT_DATASET = Path("dataset/musique/data/data/musique_ans_v1.0_dev.jsonl")
DEFAULT_PARAGRAPH_COUNT = 150


def load_paragraphs(
    path: Path = DEFAULT_DATASET, *, count: int = DEFAULT_PARAGRAPH_COUNT,
) -> list[ExtractionRequest]:
    """Select exactly ``count`` paragraphs in record/list order, preserving text.

    Blank lines retain their true source-line numbering. Paragraph indices are
    identifiers, not sort keys; the final record may be selected only in part.
    Identical text at different source positions remains separate paragraphs.
    Reject malformed records, repeated record IDs in the selected prefix and
    insufficient input rather than silently substituting other paragraphs.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("paragraph count must be a positive integer")
    checksum = compute_file_sha256(path)
    paragraphs: list[ExtractionRequest] = []
    seen_records: set[str] = set()
    for source_line, raw_line in iter_jsonl_lines(path):
        requests = build_requests_for_record(
            raw_line, source_line, str(path), checksum, "ans_dev",
        )
        record_id = requests[0].record_id
        if record_id in seen_records:
            raise ValueError(f"line {source_line}: duplicate record ID {record_id!r}")
        seen_records.add(record_id)
        paragraphs.extend(requests[:count - len(paragraphs)])
        if len(paragraphs) == count:
            return paragraphs
    raise ValueError(f"requested {count} paragraphs but dataset contains only {len(paragraphs)}")
