"""Map validated MuSiQue chunks to the SimpleKGPipeline input contract."""

from __future__ import annotations

from typing import NotRequired, TypedDict

from graphrag_pipeline.extraction.schemas import ExtractionRequest


class PluginDocumentMetadata(TypedDict):
    """Source metadata passed to SimpleKGPipeline as document metadata."""

    source_dataset: str
    source_split: str
    source_file: str
    source_sha256: str
    source_line: str
    record_id: str
    chunk_id: str
    document_id: str
    paragraph_idx: str
    title: NotRequired[str]


def build_plugin_input(
    request: ExtractionRequest,
) -> tuple[str, PluginDocumentMetadata]:
    """Build one paragraph input and its source metadata for the plugin.

    The paragraph text is passed unchanged. Gold labels, answers, question
    decomposition, and other evaluation fields are intentionally unavailable
    because the input starts from the validated Contract 2 request.
    """
    metadata: PluginDocumentMetadata = {
        "source_dataset": request.provenance.source_dataset,
        "source_split": request.provenance.source_split,
        "source_file": request.provenance.source_file,
        "source_sha256": request.provenance.source_sha256,
        "source_line": str(request.provenance.source_line),
        "record_id": request.record_id,
        "chunk_id": request.chunk_id,
        "document_id": request.document_id,
        "paragraph_idx": str(request.paragraph_idx),
    }
    if request.title is not None:
        metadata["title"] = request.title
    return request.text, metadata
