"""Summarisation providers used by ``summarize-before-store`` and compaction.

The default is extractive (frequency-weighted sentence scoring): it needs no
model and never hallucinates, which matters when the output is written back
into shared memory. An LLM-backed abstractive summariser can be slotted in
behind the same protocol.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Protocol, runtime_checkable

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "is", "are", "was",
    "were", "be", "to", "of", "in", "on", "for", "with", "as", "at", "by",
    "this", "that", "it", "we", "you", "they", "i", "from", "so", "not",
}


@runtime_checkable
class Summarizer(Protocol):
    def summarize(self, text: str, *, max_sentences: int = 5) -> str:
        ...


class ExtractiveSummarizer:
    """Selects the highest-scoring sentences, preserving original order."""

    def summarize(self, text: str, *, max_sentences: int = 5) -> str:
        text = text.strip()
        if not text:
            return ""
        sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
        if len(sentences) <= max_sentences:
            return text

        freqs: Counter[str] = Counter()
        for sentence in sentences:
            for word in _WORD_RE.findall(sentence.lower()):
                if word not in _STOPWORDS:
                    freqs[word] += 1
        if not freqs:
            return " ".join(sentences[:max_sentences])

        peak = max(freqs.values())
        scored: list[tuple[int, float]] = []
        for idx, sentence in enumerate(sentences):
            words = [w for w in _WORD_RE.findall(sentence.lower()) if w not in _STOPWORDS]
            if not words:
                scored.append((idx, 0.0))
                continue
            score = sum(freqs[w] / peak for w in words) / len(words)
            scored.append((idx, score))

        top = sorted(scored, key=lambda t: t[1], reverse=True)[:max_sentences]
        chosen = sorted(idx for idx, _ in top)
        return " ".join(sentences[i] for i in chosen)
