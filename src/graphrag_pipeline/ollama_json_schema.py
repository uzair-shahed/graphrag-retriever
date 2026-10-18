"""Ollama LLM adapter for native JSON Schema-constrained generation.

Last modified: 2026-10-18

Purpose of this file:
    Adapt Neo4j GraphRAG's Ollama wrapper so Pydantic response schemas are sent
    through Ollama's top-level ``format`` parameter for synchronous and
    asynchronous calls.

Files that use this file:
    The answer LLM factory and ``SimpleKGPipeline`` extraction factory use this
    adapter for Ollama-backed generation. Unit and Ollama integration tests
    validate its request and response behavior.

Caveats:
    JSON Schema constrains response structure, not factual correctness or
    cross-reference validity. Callers must continue validating the returned
    content against their application contract.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm import OllamaLLM
from neo4j_graphrag.llm.types import LLMResponse, LLMUsage
from neo4j_graphrag.message_history import MessageHistory
from neo4j_graphrag.types import LLMMessage
from neo4j_graphrag.utils.rate_limit import (
    async_rate_limit_handler,
    rate_limit_handler,
)


class OllamaJSONSchemaLLM(OllamaLLM):
    """Use Ollama's native JSON Schema output for GraphRAG LLM requests.

    Args:
        model_name: Ollama model tag to invoke.
        response_model: Pydantic model defining the default response schema.
        model_params: Ollama generation options, such as temperature.
        **kwargs: Ollama client configuration such as the host URL.

    Effects:
        Marks this wrapper as supporting structured output so GraphRAG can
        pass Pydantic response models to ``ainvoke``.
    """

    supports_structured_output = True

    def __init__(
        self,
        model_name: str,
        response_model: type[BaseModel],
        model_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, model_params=model_params, **kwargs)
        self.response_model = response_model
        self.response_schema = response_model.model_json_schema()

    def _schema_for(self, response_format: Any = None) -> dict[str, Any]:
        """Resolve GraphRAG's response format to an Ollama JSON Schema."""
        if response_format is None:
            return self.response_schema
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            return response_format.model_json_schema()
        if isinstance(response_format, dict):
            if response_format == {"type": "json_object"}:
                return self.response_schema
            return response_format
        raise TypeError(f"unsupported response format: {type(response_format).__name__}")

    def _messages(
        self,
        input: str | list[LLMMessage],
        message_history: list[LLMMessage] | MessageHistory | None,
        system_instruction: str | None,
    ) -> Any:
        """Convert GraphRAG input to Ollama chat messages."""
        if isinstance(input, str):
            return self.get_messages(input, message_history, system_instruction)
        if isinstance(input, list):
            return self.get_messages_v2(input)
        raise ValueError(f"Invalid input type for Ollama JSON Schema LLM: {type(input)}")

    @staticmethod
    def _response(response: Any) -> LLMResponse:
        """Convert Ollama content and optional token counts to GraphRAG's response."""
        request_tokens = response.prompt_eval_count
        response_tokens = response.eval_count
        usage = None
        if request_tokens is not None or response_tokens is not None:
            usage = LLMUsage(
                request_tokens=request_tokens,
                response_tokens=response_tokens,
                total_tokens=(request_tokens + response_tokens)
                if request_tokens is not None and response_tokens is not None
                else None,
            )
        return LLMResponse(content=response.message.content or "", usage=usage)

    @rate_limit_handler
    def invoke(
        self,
        input: str | list[LLMMessage],
        message_history: list[LLMMessage] | MessageHistory | None = None,
        system_instruction: str | None = None,
        response_format: type[BaseModel] | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a schema-constrained response synchronously.

        Args:
            input: Prompt text or GraphRAG message list.
            message_history: Optional history used with string input.
            system_instruction: Optional system instruction for string input.
            response_format: Optional per-call Pydantic model or JSON Schema.
            **kwargs: Additional parameters accepted by Ollama's chat API.

        Returns:
            GraphRAG response content and token usage, when reported.
        """
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=self._messages(input, message_history, system_instruction),
                format=self._schema_for(response_format),
                options=self.model_params.get("options", {}),
                **kwargs,
            )
        except self.ollama.ResponseError as error:
            raise LLMGenerationError(error) from error
        return self._response(response)

    @async_rate_limit_handler
    async def ainvoke(
        self,
        input: str | list[LLMMessage],
        message_history: list[LLMMessage] | MessageHistory | None = None,
        system_instruction: str | None = None,
        response_format: type[BaseModel] | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a schema-constrained response asynchronously.

        Args:
            input: Prompt text or GraphRAG message list.
            message_history: Optional history used with string input.
            system_instruction: Optional system instruction for string input.
            response_format: Optional per-call Pydantic model or JSON Schema.
            **kwargs: Additional parameters accepted by Ollama's chat API.

        Returns:
            GraphRAG response content and token usage, when reported.
        """
        try:
            response = await self.async_client.chat(
                model=self.model_name,
                messages=self._messages(input, message_history, system_instruction),
                format=self._schema_for(response_format),
                options=self.model_params.get("options", {}),
                **kwargs,
            )
        except self.ollama.ResponseError as error:
            raise LLMGenerationError(error) from error
        return self._response(response)