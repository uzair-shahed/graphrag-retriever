"""Model content and local provenance binding using the shared extraction types."""

from pydantic import BaseModel, ConfigDict

from graphrag_pipeline.extraction.schemas import (
    ExtractionRequest,
    ExtractionResult,
    Mention,
    Triple,
    validate_result_for_request,
)


class ParagraphExtraction(BaseModel):
    """Provider output using the existing mention and directed-triple schemas.

    Source identity/text/provenance are attached locally after generation so
    providers need only the title and paragraph body. Both providers will use
    ``model_json_schema()`` from this class for constrained generation.
    """

    model_config = ConfigDict(extra="forbid")

    mentions: list[Mention]
    triples: list[Triple]


def model_input(request: ExtractionRequest) -> dict[str, str | None]:
    """Return only the chosen title and text, with no record or gold fields."""
    return {"title": request.title, "text": request.text}


def validate_extraction_response(request: ExtractionRequest, content: str) -> ExtractionResult:
    """Parse provider JSON, bind local identity, and check references and spans.

    Invalid JSON, extra fields, invalid scalar types, unknown mention IDs, or
    evidence not matching the original paragraph raise validation errors. Empty
    mention/triple arrays are valid; a failed call must be tracked separately.
    """
    extracted = ParagraphExtraction.model_validate_json(content)
    result = ExtractionResult(
        chunk_id=request.chunk_id,
        record_id=request.record_id,
        chunk_text=request.text,
        provenance=request.provenance,
        mentions=extracted.mentions,
        triples=extracted.triples,
    )
    return validate_result_for_request(request, result)
