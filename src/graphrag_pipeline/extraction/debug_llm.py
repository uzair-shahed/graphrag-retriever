"""LLM helpers for preserving invalid GraphRAG responses during debugging."""

from __future__ import annotations

from typing import Any

from neo4j_graphrag.llm import OpenAILLM


class DebuggableOpenAILLM(OpenAILLM):
    """Capture the most recent raw response before GraphRAG validates it."""

    last_response: str | None = None

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke OpenAI-compatible LLM and retain its raw response content."""
        response = await super().ainvoke(*args, **kwargs)
        self.last_response = response.content
        return response

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the synchronous API and retain its raw response content."""
        response = super().invoke(*args, **kwargs)
        self.last_response = response.content
        return response