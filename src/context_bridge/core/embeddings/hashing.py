"""A deterministic, dependency-free embedder based on feature hashing.

This is the default provider so the whole system runs (and is fully testable)
without downloading model weights or hitting the network. It produces a signed
bag-of-words dense vector plus a term-frequency sparse vector, which is enough
for lexical-overlap similarity to behave sensibly. Swap in ``fastembed`` for
production-grade semantic quality.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from context_bridge.core.models import SparseVector

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SPARSE_SPACE = 2**20


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _stable_hash(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


class HashingEmbedder:
    """Feature-hashing embedder (see module docstring)."""

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    @property
    def dense_dim(self) -> int:
        return self._dim

    @property
    def supports_sparse(self) -> bool:
        return True

    def _dense_one(self, text: str) -> list[float]:
        vec = np.zeros(self._dim, dtype=np.float32)
        for token in _tokenize(text):
            h = _stable_hash(token)
            bucket = h % self._dim
            sign = 1.0 if (h >> 31) & 1 else -1.0
            vec[bucket] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def _sparse_one(self, text: str) -> SparseVector:
        counts: dict[int, float] = {}
        for token in _tokenize(text):
            idx = _stable_hash(token) % _SPARSE_SPACE
            counts[idx] = counts.get(idx, 0.0) + 1.0
        if not counts:
            return SparseVector()
        indices = sorted(counts)
        # Sub-linear term weighting, mirroring BM25-style saturation.
        values = [1.0 + np.log(counts[i]) for i in indices]
        return SparseVector(indices=indices, values=[float(v) for v in values])

    def embed_dense(self, texts: list[str]) -> list[list[float]]:
        return [self._dense_one(t) for t in texts]

    def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        return [self._sparse_one(t) for t in texts]

    def embed_query_dense(self, text: str) -> list[float]:
        return self._dense_one(text)

    def embed_query_sparse(self, text: str) -> SparseVector:
        return self._sparse_one(text)
