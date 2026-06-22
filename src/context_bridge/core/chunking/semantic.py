"""Embedding-driven semantic chunker.

Sentences are embedded and a new chunk boundary is placed wherever the
similarity between consecutive sentences drops below an adaptive threshold
(derived from the distribution of distances). Groups that still exceed the
token budget are handed to the recursive chunker as a safety net.
"""

from __future__ import annotations

import re

import numpy as np

from context_bridge.core.chunking.base import pack_pieces, to_chunks
from context_bridge.core.chunking.recursive import RecursiveChunker
from context_bridge.core.embeddings.base import Embedder
from context_bridge.core.models import Chunk
from context_bridge.tokenizer import count_tokens

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class SemanticChunker:
    """See module docstring."""

    def __init__(
        self,
        embedder: Embedder,
        *,
        chunk_size: int = 256,
        overlap: int = 32,
        percentile: float = 90.0,
    ) -> None:
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.percentile = percentile
        self._fallback = RecursiveChunker(chunk_size=chunk_size, overlap=overlap)

    def chunk(self, text: str, *, parent_id: str | None = None) -> list[Chunk]:
        text = text.strip()
        if not text:
            return []
        sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
        if len(sentences) <= 1:
            return self._fallback.chunk(text, parent_id=parent_id)

        groups = self._group_by_similarity(sentences)

        packed: list[str] = []
        for group in groups:
            joined = " ".join(group)
            if count_tokens(joined) <= self.chunk_size:
                packed.append(joined)
            else:
                packed.extend(
                    pack_pieces(group, size=self.chunk_size, overlap=self.overlap)
                )
        return to_chunks(packed, full_text=text, parent_id=parent_id)

    def _group_by_similarity(self, sentences: list[str]) -> list[list[str]]:
        vecs = np.asarray(self.embedder.embed_dense(sentences), dtype=np.float32)
        sims = [float(np.dot(vecs[i], vecs[i + 1])) for i in range(len(vecs) - 1)]
        if not sims:
            return [sentences]
        # Break where similarity is unusually low (a topical shift).
        threshold = float(np.percentile(sims, 100.0 - self.percentile))

        groups: list[list[str]] = [[sentences[0]]]
        for i, sim in enumerate(sims):
            if sim < threshold:
                groups.append([])
            groups[-1].append(sentences[i + 1])
        return groups
