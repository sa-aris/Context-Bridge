"""OpenAI (and OpenAI-compatible) embeddings.

Works with the official API or any compatible endpoint (Azure OpenAI, vLLM,
LiteLLM, ...) via ``base_url``. Loads the client lazily so importing this module
never requires the ``openai`` package unless the provider is actually used.
"""

from __future__ import annotations

from functools import cached_property

from context_bridge.core.models import SparseVector

# Known output dimensionalities so the vector store can be sized without a call.
_MODEL_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder:
    """Dense embeddings via the OpenAI embeddings API."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        api_key: str = "",
        base_url: str = "",
        dim: int | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self._dim = _MODEL_DIMS.get(model) or dim or 1536

    @property
    def dense_dim(self) -> int:
        return self._dim

    @property
    def supports_sparse(self) -> bool:
        return False

    @cached_property
    def _client(self):  # pragma: no cover - requires the openai package + network
        from openai import OpenAI

        return OpenAI(api_key=self.api_key or None, base_url=self.base_url or None)

    def embed_dense(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - network
        resp = self._client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in resp.data]

    def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        return [SparseVector() for _ in texts]

    def embed_query_dense(self, text: str) -> list[float]:  # pragma: no cover - network
        return self.embed_dense([text])[0]

    def embed_query_sparse(self, text: str) -> SparseVector:
        return SparseVector()
