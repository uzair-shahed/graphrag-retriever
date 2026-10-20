"""Explicit RAGAS clients: never fall back to an unconfigured OpenAI service."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass(frozen=True)
class EvaluatorSettings:
    provider: Literal["ollama", "openrouter"]
    llm_model: str
    embedding_model: str
    base_url: str
    api_key: str = field(repr=False)
    max_tokens: int = 4096
    timeout: float = 180
    retries: int = 2

    def __post_init__(self):
        if self.provider not in {"ollama", "openrouter"}:
            raise ValueError("RAGAS_PROVIDER must be ollama or openrouter")
        if not all((self.llm_model, self.embedding_model, self.base_url, self.api_key)):
            raise ValueError("evaluator model, embedding, endpoint and credentials are required")
        if self.max_tokens < 1 or self.timeout <= 0 or self.retries < 0:
            raise ValueError("invalid evaluator token/timeout/retry settings")

    def public(self):
        return {k: v for k, v in asdict(self).items() if k != "api_key"}

    @classmethod
    def from_env(cls, provider=None):
        provider = (provider or os.getenv("RAGAS_PROVIDER", os.getenv("GRAPH_RAG_PROVIDER", "openrouter"))).strip().lower()
        if provider not in {"ollama", "openrouter"}:
            raise ValueError("RAGAS_PROVIDER must be ollama or openrouter")
        prefix = "OLLAMA" if provider == "ollama" else "OPENROUTER"
        base = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/") + "/v1" if provider == "ollama" else os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        return cls(
            provider=provider,
            llm_model=os.getenv("RAGAS_LLM_MODEL") or os.getenv(f"{prefix}_LLM_MODEL", "qwen3:4b" if provider == "ollama" else "openai/gpt-4o-mini"),
            embedding_model=os.getenv("RAGAS_EMBEDDING_MODEL") or os.getenv(f"{prefix}_EMBEDDING_MODEL", "nomic-embed-text" if provider == "ollama" else "openai/text-embedding-3-small"),
            base_url=os.getenv("RAGAS_BASE_URL") or base,
            api_key="ollama" if provider == "ollama" else os.getenv("OPENROUTER_API_KEY", ""),
            max_tokens=int(os.getenv("RAGAS_MAX_TOKENS", "4096")),
            timeout=float(os.getenv("RAGAS_TIMEOUT_SECONDS", "180")),
            retries=int(os.getenv("RAGAS_RETRIES", "2")),
        )


def build_metrics(settings: EvaluatorSettings):
    # RAGAS analytics are unnecessary for local evaluation. Set before imports.
    os.environ["RAGAS_DO_NOT_TRACK"] = "true"
    from openai import AsyncOpenAI
    from ragas.embeddings import OpenAIEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerRelevancy, ContextRelevance, Faithfulness

    client = AsyncOpenAI(api_key=settings.api_key, base_url=settings.base_url,
                         timeout=settings.timeout, max_retries=settings.retries)
    kwargs = {"temperature": 0, "max_tokens": settings.max_tokens, "max_retries": settings.retries + 1}
    if settings.provider == "openrouter":
        kwargs["extra_body"] = {"provider": {"require_parameters": True, "allow_fallbacks": False}}
    llm = llm_factory(settings.llm_model, provider="openai", client=client, **kwargs)
    embeddings = OpenAIEmbeddings(client=client, model=settings.embedding_model)
    return {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_relevance": ContextRelevance(llm=llm, max_retries=settings.retries + 1),
    }, client
