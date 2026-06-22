"""Pluggable embedding providers behind a common protocol."""

from __future__ import annotations

from context_bridge.config import Settings
from context_bridge.core.embeddings.base import Embedder
from context_bridge.core.embeddings.hashing import HashingEmbedder


def build_embedder(settings: Settings) -> Embedder:
    """Construct the configured embedder.

    ``hashing`` is dependency-free and deterministic (ideal for tests and
    offline use); ``fastembed`` loads local ONNX models for production quality.
    """
    provider = settings.embed_provider.lower()
    if provider == "hashing":
        return HashingEmbedder(dim=settings.embed_dim)
    if provider == "fastembed":
        from context_bridge.core.embeddings.fastembed_embedder import FastEmbedEmbedder

        return FastEmbedEmbedder(
            dense_model=settings.embed_dense_model,
            sparse_model=settings.embed_sparse_model,
        )
    raise ValueError(f"Unknown embed_provider: {settings.embed_provider!r}")


__all__ = ["Embedder", "HashingEmbedder", "build_embedder"]
