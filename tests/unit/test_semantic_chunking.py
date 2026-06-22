from __future__ import annotations

from context_bridge.core.chunking.semantic import SemanticChunker
from context_bridge.core.embeddings.hashing import HashingEmbedder


def test_semantic_chunker_groups_sentences():
    embedder = HashingEmbedder(dim=128)
    chunker = SemanticChunker(embedder, chunk_size=64, overlap=8)
    text = (
        "The deployment pipeline builds the image. It then runs the test suite. "
        "Separately, the marketing team planned a launch event. "
        "Catering was booked for two hundred guests."
    )

    chunks = chunker.chunk(text)

    assert len(chunks) >= 1
    assert all(c.parent_text == text for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_single_sentence_falls_back():
    embedder = HashingEmbedder(dim=128)
    chunker = SemanticChunker(embedder, chunk_size=64, overlap=8)
    chunks = chunker.chunk("Just one sentence here.")
    assert len(chunks) == 1


def test_empty_text_yields_no_chunks():
    embedder = HashingEmbedder(dim=128)
    assert SemanticChunker(embedder).chunk("  ") == []
