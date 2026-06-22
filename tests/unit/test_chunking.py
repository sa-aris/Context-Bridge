from __future__ import annotations

from context_bridge.core.chunking.recursive import RecursiveChunker
from context_bridge.tokenizer import count_tokens


def test_short_text_is_single_chunk():
    chunker = RecursiveChunker(chunk_size=64, overlap=8)
    chunks = chunker.chunk("A short note about the deployment plan.")
    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert chunks[0].parent_id == chunks[0].parent_id  # stable id assigned


def test_long_text_splits_within_budget():
    chunker = RecursiveChunker(chunk_size=32, overlap=4)
    paragraphs = [f"Paragraph {i} discusses topic {i} in some detail." * 4 for i in range(10)]
    text = "\n\n".join(paragraphs)

    chunks = chunker.chunk(text)

    assert len(chunks) > 1
    for chunk in chunks:
        assert count_tokens(chunk.text) <= 32 + 4  # budget plus a small tolerance


def test_chunks_share_parent_and_carry_full_text():
    chunker = RecursiveChunker(chunk_size=16, overlap=2)
    text = " ".join(f"word{i}" for i in range(200))

    chunks = chunker.chunk(text)
    parent_ids = {c.parent_id for c in chunks}

    assert len(parent_ids) == 1
    assert all(c.parent_text == text for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_empty_text_yields_no_chunks():
    assert RecursiveChunker().chunk("   \n  ") == []
