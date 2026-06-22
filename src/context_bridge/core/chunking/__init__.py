"""Chunking strategies behind a common protocol."""

from __future__ import annotations

from context_bridge.config import Settings
from context_bridge.core.chunking.base import Chunker
from context_bridge.core.chunking.recursive import RecursiveChunker


def build_chunker(settings: Settings, embedder=None) -> Chunker:
    """Construct the recursive token-aware chunker (the safe default)."""
    return RecursiveChunker(
        chunk_size=settings.chunk_size_tokens,
        overlap=settings.chunk_overlap_tokens,
    )


__all__ = ["Chunker", "RecursiveChunker", "build_chunker"]
