"""Vector-store providers behind a common protocol."""

from __future__ import annotations

from context_bridge.config import Settings
from context_bridge.core.vectorstore.base import VectorStore
from context_bridge.core.vectorstore.qdrant_store import QdrantStore


def build_vector_store(settings: Settings, dim: int, *, supports_sparse: bool) -> VectorStore:
    """Construct and initialise the configured vector store."""
    store = QdrantStore.from_url(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection=settings.qdrant_collection,
        dim=dim,
        with_sparse=supports_sparse,
    )
    store.ensure_collection()
    return store


__all__ = ["VectorStore", "QdrantStore", "build_vector_store"]
