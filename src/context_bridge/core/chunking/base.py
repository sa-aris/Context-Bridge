"""The chunker contract and shared helpers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from context_bridge.core.models import Chunk, new_id
from context_bridge.tokenizer import count_tokens


@runtime_checkable
class Chunker(Protocol):
    """Splits a document into retrieval-sized chunks.

    Implementations attach a shared ``parent_id`` and the full ``parent_text``
    to every chunk so the retrieval layer can apply the small-to-big strategy:
    match on a small chunk, then expand to its parent for fuller context.
    """

    def chunk(self, text: str, *, parent_id: str | None = None) -> list[Chunk]: ...


def pack_pieces(pieces: list[str], *, size: int, overlap: int, sep: str = " ") -> list[str]:
    """Greedily merge ``pieces`` into chunks of ~``size`` tokens with overlap.

    When a chunk fills up, the next one is seeded with a trailing window of the
    previous pieces (~``overlap`` tokens) so context is not severed at the seam.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        pt = count_tokens(piece)
        if current and current_tokens + pt > size:
            chunks.append(sep.join(current))
            tail: list[str] = []
            tail_tokens = 0
            for prev in reversed(current):
                ptoks = count_tokens(prev)
                if tail_tokens + ptoks > overlap:
                    break
                tail.insert(0, prev)
                tail_tokens += ptoks
            current = tail
            current_tokens = tail_tokens
        current.append(piece)
        current_tokens += pt

    if current:
        chunks.append(sep.join(current))
    return chunks


def to_chunks(texts: list[str], *, full_text: str, parent_id: str | None) -> list[Chunk]:
    """Wrap raw chunk strings into :class:`Chunk` objects sharing a parent."""
    pid = parent_id or new_id()
    return [
        Chunk(text=t, index=i, parent_id=pid, parent_text=full_text) for i, t in enumerate(texts)
    ]
