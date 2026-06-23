"""Cohere embeddings (multilingual-capable, v3 input-type aware)."""

from __future__ import annotations

from functools import cached_property

from context_bridge.core.models import SparseVector

_MODEL_DIMS = {
    "embed-english-v3.0": 1024,
    "embed-multilingual-v3.0": 1024,
    "embed-english-light-v3.0": 384,
    "embed-multilingual-light-v3.0": 384,
}


class CohereEmbedder:
    """Dense embeddings via the Cohere embed API."""

    def __init__(
        self,
        model: str = "embed-english-v3.0",
        *,
        api_key: str = "",
        dim: int | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self._dim = _MODEL_DIMS.get(model) or dim or 1024

    @property
    def dense_dim(self) -> int:
        return self._dim

    @property
    def supports_sparse(self) -> bool:
        return False

    @cached_property
    def _client(self):  # pragma: no cover - requires the cohere package + network
        import cohere

        return cohere.Client(self.api_key or None)

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:  # pragma: no cover
        resp = self._client.embed(texts=texts, model=self.model, input_type=input_type)
        return [list(v) for v in resp.embeddings]

    def embed_dense(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - network
        return self._embed(texts, "search_document")

    def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        return [SparseVector() for _ in texts]

    def embed_query_dense(self, text: str) -> list[float]:  # pragma: no cover - network
        return self._embed([text], "search_query")[0]

    def embed_query_sparse(self, text: str) -> SparseVector:
        return SparseVector()
