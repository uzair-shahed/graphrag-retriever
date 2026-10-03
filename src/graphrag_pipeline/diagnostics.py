"""Format provider failures without exposing credentials.

This module is used by the fixture and batch runners to preserve useful
exception-chain details from GraphRAG, the OpenAI SDK, and OpenRouter.
Provider response bodies are inspected when available; secrets are redacted.
"""

from __future__ import annotations

import json
import re
from typing import Any


_SECRET_PATTERN = re.compile(
    r"(?i)(bearer\s+|sk-or-v1-[A-Za-z0-9]+|api[_-]?key[\"']?\s*[:=]\s*[\"']?)[^\s,\"'}]+"
)


def _redact(value: str) -> str:
    """Redact bearer tokens and common API-key values from diagnostic text."""
    return _SECRET_PATTERN.sub(lambda match: f"{match.group(1)}[REDACTED]", value)


def _response_details(error: BaseException) -> dict[str, Any] | None:
    """Extract a JSON-safe provider response body from an SDK exception."""
    response = getattr(error, "response", None)
    body = getattr(response, "text", None) if response is not None else None
    if body is None and response is not None:
        body = getattr(response, "content", None)
    if isinstance(body, bytes):
        body = body.decode("utf-8", errors="replace")
    if isinstance(body, str):
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return {"response_body": _redact(body)}
        return {"response_body": json.loads(_redact(json.dumps(parsed)))}

    body = getattr(error, "body", None)
    if isinstance(body, dict):
        return {"response_body": json.loads(_redact(json.dumps(body)))}
    return None


def exception_diagnostics(error: BaseException) -> dict[str, Any]:
    """Return a redacted, JSON-safe summary of an exception and its causes.

    Args:
        error: The outer exception raised by the application or GraphRAG.

    Returns:
        A dictionary containing the exception chain and provider response data.
    """
    chain: list[dict[str, Any]] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        item: dict[str, Any] = {
            "type": type(current).__name__,
            "message": _redact(str(current)),
        }
        details = _response_details(current)
        if details:
            item.update(details)
        chain.append(item)
        current = current.__cause__ or current.__context__

    return {
        "error_type": type(error).__name__,
        "error": _redact(str(error)),
        "exception_chain": chain,
    }