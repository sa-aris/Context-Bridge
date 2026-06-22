"""Local, open-model embeddings via FastEmbed (ONNX runtime, no heavy deps).

Loads models lazily on first use so importing this module never triggers a
download. Produces both dense and sparse vectors, enabling true hybrid search.
"""

from __future__ import annotations

from functools import cached_property

from context_bridge.core.models import SparseVector


class FastEmbedEmbedder:
    """Embedder backed by FastEmbed dense + sparse models."""

    def __init__(
        self,
        dense_model: str = "BAAI/bge-small-en-v1.5",
        sparse_model: str = "Qdrant/bm25",
    ) -> None:
        self._dense_model_name = dense_model
        self._sparse_model_name = sparse_model

    @cached_property
    def _dense(self):  # pragma: no cover - requires model download
        from fastembed import TextEmbedding

        return TextEmbedding(model_name=self._dense_model_name)

    @cached_property
    def _sparse(self):  # pragma: no cover - requires model download
        from fastembed import SparseTextEmbedding

        return SparseTextEmbedding(model_name=self._sparse_model_name)

    @cached_property
    def dense_dim(self) -> int:  # pragma: no cover - requires model download
        return len(next(iter(self._dense.embed(["dimension probe"]))))

    @property
    def supports_sparse(self) -> bool:
        return True

    def embed_dense(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        return [vec.tolist() for vec in self._dense.embed(texts)]

    def embed_sparse(self, texts: list[str]) -> list[SparseVector]:  # pragma: no cover
        out: list[SparseVector] = []
        for sv in self._sparse.embed(texts):
            out.append(
                SparseVector(
                    indices=[int(i) for i in sv.indices],
                    values=[float(v) for v in sv.values],
                )
            )
        return out

    def embed_query_dense(self, text: str) -> list[float]:  # pragma: no cover
        return self.embed_dense([text])[0]

    def embed_query_sparse(self, text: str) -> SparseVector:  # pragma: no cover
        return self.embed_sparse([text])[0]
