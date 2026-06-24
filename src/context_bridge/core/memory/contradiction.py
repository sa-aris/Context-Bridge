"""Detect contradictions between memories (truth-maintenance).

When two memories are about the *same thing* (high semantic similarity) but
disagree, the shared pool should flag it rather than silently hold both beliefs.
:class:`HeuristicDetector` is deterministic: it fires on negation-polarity flips
and numeric mismatches between topically-overlapping statements. Swap in an
NLI/LLM detector behind :class:`ContradictionDetector` for nuance.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

_WORD_RE = re.compile(r"[a-z0-9]+")
_NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_NEGATIONS = {
    "not",
    "no",
    "never",
    "none",
    "cannot",
    "can't",
    "won't",
    "don't",
    "doesn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "shouldn't",
    "without",
    "fails",
    "failed",
    "disabled",
    "false",
    "off",
}
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
}


@runtime_checkable
class ContradictionDetector(Protocol):
    def is_contradiction(self, a: str, b: str) -> bool: ...


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOP and w not in _NEGATIONS}


class NullDetector:
    def is_contradiction(self, a: str, b: str) -> bool:
        return False


class HeuristicDetector:
    """Negation-flip and numeric-mismatch contradiction heuristic."""

    def __init__(self, min_overlap: float = 0.4) -> None:
        self.min_overlap = min_overlap

    def is_contradiction(self, a: str, b: str) -> bool:
        wa, wb = _content_words(a), _content_words(b)
        if not wa or not wb:
            return False
        overlap = len(wa & wb) / min(len(wa), len(wb))
        if overlap < self.min_overlap:
            return False  # different topics — not a contradiction, just unrelated

        neg_a = bool(_NEGATIONS & set(_WORD_RE.findall(a.lower())))
        neg_b = bool(_NEGATIONS & set(_WORD_RE.findall(b.lower())))
        if neg_a != neg_b:
            return True  # one asserts, the other negates the same topic

        nums_a, nums_b = set(_NUM_RE.findall(a)), set(_NUM_RE.findall(b))
        return bool(nums_a and nums_b and nums_a != nums_b)


def build_detector(settings) -> ContradictionDetector:
    return HeuristicDetector() if settings.detect_contradictions else NullDetector()
