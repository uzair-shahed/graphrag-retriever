"""Minimal model output and local provenance binding using shared extraction types."""

import copy
import json

from pydantic import BaseModel, ConfigDict

from graphrag_pipeline.extraction.schemas import (
    ExtractionRequest, ExtractionResult, Mention, Triple, validate_result_for_request,
)


class ParagraphExtraction(BaseModel):
    """Entities and directed relationships; source identity is attached locally."""

    model_config = ConfigDict(extra="forbid")

    mentions: list[Mention]
    triples: list[Triple]


def model_input(request: ExtractionRequest) -> dict[str, str | None]:
    """Return title and body only, without record or gold fields."""
    return {"title": request.title, "text": request.text}


def project_legacy_payload(data):
    """Remove only retired v1 fields; keep every other value for strict validation."""
    data = copy.deepcopy(data)
    if isinstance(data, dict):
        for name in ("mentions", "triples"):
            items = data.get(name)
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict):
                    item.pop("evidence_span", None)
                    if name == "triples":
                        item.pop("confidence", None)
    return data


def _reject_constant(value):
    raise ValueError(f"invalid JSON number {value}")


def validate_extraction_response(
    request: ExtractionRequest, content: str, *, response_format_version: int = 2
) -> ExtractionResult:
    """Validate model output and references, then bind source identity/provenance.

    Legacy projection is used only for explicitly migrated, archived responses.
    Newly generated output must match the minimal schema without extra fields.
    """
    if response_format_version == 1:
        data = json.loads(content, parse_constant=_reject_constant)
        content = json.dumps(project_legacy_payload(data), allow_nan=False)
    elif response_format_version != 2:
        raise ValueError("unsupported response format version")
    extracted = ParagraphExtraction.model_validate_json(content)
    result = ExtractionResult(
        chunk_id=request.chunk_id, record_id=request.record_id,
        chunk_text=request.text, provenance=request.provenance,
        mentions=extracted.mentions, triples=extracted.triples,
    )
    return validate_result_for_request(request, result)
