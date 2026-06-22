"""The embedding provider contract."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from context_bridge.core.models import SparseVector


@runtime_checkable
class Embedder(Protocol):
    """Produces dense and (optionally) sparse vectors for text.

    Dense vectors power semantic similarity; sparse vectors power lexical /
    keyword matching. Combining both via fusion is what gives hybrid search
    its precision.
    """

    @property
    def dense_dim(self) -> int:
        """Dimensionality of the dense vectors produced by this embedder."""
        ...

    @property
    def supports_sparse(self) -> bool:
        """Whether :meth:`embed_sparse` returns meaningful vectors."""
        ...

    def embed_dense(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts`` into dense vectors."""
        ...

    def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        """Embed ``texts`` into sparse vectors (may be empty if unsupported)."""
        ...

    def embed_query_dense(self, text: str) -> list[float]:
        """Embed a single query string into a dense vector."""
        ...

    def embed_query_sparse(self, text: str) -> SparseVector:
        """Embed a single query string into a sparse vector."""
        ...
