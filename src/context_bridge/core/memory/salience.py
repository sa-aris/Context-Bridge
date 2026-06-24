"""Salience scoring: decide which conversational turns are worth keeping.

Humans don't memorize every sentence — they retain what they kept returning to,
what was emphasized, what carried substance. :class:`SalienceScorer` approximates
that deterministically over a stream of turns so the system can *distill* a noisy
conversation down to the few memories worth carrying into future sessions.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "as",
    "at",
    "by",
    "this",
    "that",
    "it",
    "and",
    "or",
    "its",
    "we",
    "you",
    "they",
    "i",
    "from",
    "so",
    "not",
    "but",
    "if",
    "then",
    "have",
    "has",
    "do",
    "does",
    "can",
    "will",
    "just",
    "ok",
    "okay",
    "yeah",
    "yes",
    "no",
}
_EMPHASIS = (
    "important",
    "remember",
    "note that",
    "key ",
    "critical",
    "decision",
    "decided",
    "must ",
    "always",
    "never",
    "don't forget",
    "do not forget",
    "action item",
    "todo",
    "takeaway",
    "in summary",
)


@dataclass(slots=True)
class ScoredTurn:
    text: str
    score: float


class SalienceScorer:
    """Score and select the most salient turns from a conversation."""

    def __init__(self, *, min_score: float = 1.0) -> None:
        self.min_score = min_score

    def distill(
        self, turns: list[str], *, max_promote: int = 5, min_score: float | None = None
    ) -> list[ScoredTurn]:
        texts = [t.strip() for t in turns if t and t.strip()]
        if not texts:
            return []

        per_turn_words: list[set[str]] = []
        doc_freq: Counter[str] = Counter()
        for text in texts:
            words = {w for w in _WORD_RE.findall(text.lower()) if w not in _STOP and len(w) > 2}
            per_turn_words.append(words)
            doc_freq.update(words)

        scored: list[ScoredTurn] = []
        for text, words in zip(texts, per_turn_words, strict=True):
            # Returned-to topics: words that recur across other turns.
            recurrence = sum(doc_freq[w] - 1 for w in words)
            lower = text.lower()
            emphasis = sum(1 for marker in _EMPHASIS if marker in lower)
            emphasis += lower.count("!")
            emphasis += sum(1 for tok in text.split() if len(tok) > 1 and tok.isupper())
            substance = min(len(words) / 8.0, 1.5)
            score = 0.5 * recurrence + 1.5 * emphasis + substance
            scored.append(ScoredTurn(text=text, score=round(score, 3)))

        threshold = self.min_score if min_score is None else min_score
        scored.sort(key=lambda s: s.score, reverse=True)
        return [s for s in scored if s.score >= threshold][:max_promote]
