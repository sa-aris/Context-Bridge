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


class CohereReranker:
    """Cross-encoder reranking via the Cohere rerank API."""

    def __init__(self, model: str = "rerank-english-v3.0", *, api_key: str = "") -> None:
        self._model_name = model
        self.api_key = api_key

    @cached_property
    def _client(self):  # pragma: no cover - requires cohere + network
        import cohere

        return cohere.Client(self.api_key or None)

    def rerank(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:  # pragma: no cover - network
        if not chunks:
            return chunks
        resp = self._client.rerank(
            model=self._model_name,
            query=query,
            documents=[c.content for c in chunks],
            top_n=len(chunks),
        )
        reordered: list[RetrievedChunk] = []
        for result in resp.results:
            chunk = chunks[result.index]
            chunk.score = float(result.relevance_score)
            reordered.append(chunk)
        return reordered


def build_reranker(settings: Settings) -> Reranker:
    """Construct the configured reranker."""
    provider = settings.rerank_provider.lower()
    if provider == "identity":
        return IdentityReranker()
    if provider == "fastembed":
        return FastEmbedReranker(model=settings.rerank_model)
    if provider == "cohere":
        return CohereReranker(model=settings.rerank_model, api_key=settings.cohere_api_key)
    raise ValueError(f"Unknown rerank_provider: {settings.rerank_provider!r}")
