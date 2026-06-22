"""The vector-store contract.

Keeping this abstract is what prevents vendor lock-in: the rest of the system
depends only on these methods, so Qdrant can be swapped for Pinecone, Chroma,
etc. by implementing the same protocol.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from context_bridge.core.models import MemoryRecord, RetrievedChunk, SparseVector


@runtime_checkable
class VectorStore(Protocol):
    """Persists embedded memories and answers hybrid similarity queries."""

    def ensure_collection(self) -> None:
        """Create the backing collection if it does not already exist."""
        ...

    def upsert(self, records: list[MemoryRecord]) -> None:
        """Insert or update a batch of fully embedded records."""
        ...

    def hybrid_search(
        self,
        *,
        dense: list[float],
        sparse: SparseVector | None,
        limit: int,
        namespace: str | None = None,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]:
        """Return candidates fusing dense and sparse similarity."""
        ...

    def get(self, record_id: str) -> RetrievedChunk | None:
        """Fetch a single stored record by id."""
        ...

    def delete(self, record_ids: list[str]) -> None:
        """Remove records by id."""
        ...

    def sweep_expired(self, *, batch_size: int = 256) -> int:
        """Physically delete records whose TTL has elapsed; return the count."""
        ...
