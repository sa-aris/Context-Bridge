"""Token-aware recursive chunker.

Splits on progressively finer separators (paragraphs → lines → sentences →
words) until every piece fits the budget, then greedily packs the pieces back
into overlapping chunks. This keeps semantic boundaries intact far better than
a fixed character window.
"""

from __future__ import annotations

from context_bridge.core.chunking.base import pack_pieces, to_chunks
from context_bridge.core.models import Chunk
from context_bridge.tokenizer import count_tokens

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class RecursiveChunker:
    """See module docstring."""

    def __init__(self, chunk_size: int = 256, overlap: int = 32) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, *, parent_id: str | None = None) -> list[Chunk]:
        text = text.strip()
        if not text:
            return []
        pieces = self._split(text, _SEPARATORS)
        packed = pack_pieces(pieces, size=self.chunk_size, overlap=self.overlap)
        return to_chunks(packed, full_text=text, parent_id=parent_id)

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if count_tokens(text) <= self.chunk_size:
            return [text]
        if not separators:
            return [text]

        sep = separators[-1]
        for candidate in separators:
            if candidate == "" or candidate in text:
                sep = candidate
                break

        rest = separators[separators.index(sep) + 1 :]
        parts = list(text) if sep == "" else text.split(sep)

        out: list[str] = []
        for part in parts:
            if not part:
                continue
            if count_tokens(part) <= self.chunk_size:
                out.append(part)
            else:
                out.extend(self._split(part, rest))
        return out
