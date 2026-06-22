"""Reranking providers.

A cross-encoder rerank step is the single biggest precision win in a RAG
pipeline: it scores each (query, chunk) pair jointly instead of comparing
independent vectors. The identity reranker keeps the fusion order and is used
when no model is configured (e.g. offline / tests).
"""

from __future__ import annotations

from functools import cached_property
from typing import Protocol, runtime_checkable

from context_bridge.config import Settings
from context_bridge.core.models import RetrievedChunk


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Return ``chunks`` re-scored and sorted by relevance to ``query``."""
        ...


class IdentityReranker:
    """No-op reranker that preserves the incoming (fusion) order."""

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return chunks


class FastEmbedReranker:
    """Cross-encoder reranker backed by FastEmbed."""

    def __init__(self, model: str = "Xenova/ms-marco-MiniLM-L-6-v2") -> None:
        self._model_name = model

    @cached_property
    def _model(self):  # pragma: no cover - requires model download
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        return TextCrossEncoder(model_name=self._model_name)

    def rerank(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:  # pragma: no cover - requires model download
        if not chunks:
            return chunks
        scores = list(self._model.rerank(query, [c.content for c in chunks]))
        for chunk, score in zip(chunks, scores, strict=True):
            chunk.score = float(score)
        return sorted(chunks, key=lambda c: c.score, reverse=True)


def build_reranker(settings: Settings) -> Reranker:
    """Construct the configured reranker."""
    provider = settings.rerank_provider.lower()
    if provider == "identity":
        return IdentityReranker()
    if provider == "fastembed":
        return FastEmbedReranker(model=settings.rerank_model)
    raise ValueError(f"Unknown rerank_provider: {settings.rerank_provider!r}")
