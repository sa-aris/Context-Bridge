"""Redact secrets and PII before content enters the shared pool.

A shared memory pool is the wrong place for raw API keys, card numbers or
personal data. :class:`RegexRedactor` masks the common, high-confidence patterns
deterministically (no model, no network). Swap in a richer detector behind the
:class:`Redactor` protocol when you need NER-grade coverage.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # Common secret/token shapes: long base64-ish strings and provider prefixes.
    ("SECRET", re.compile(r"\b(?:sk|pk|ghp|xox[baprs])[-_][A-Za-z0-9-_]{16,}\b")),
    ("APIKEY", re.compile(r"\b[A-Za-z0-9_-]{32,}\b")),
    ("PHONE", re.compile(r"\b\+?\d[\d ().-]{7,}\d\b")),
]


@runtime_checkable
class Redactor(Protocol):
    def redact(self, text: str) -> str: ...


class NullRedactor:
    """No-op redactor (default when redaction is disabled)."""

    def redact(self, text: str) -> str:
        return text


class RegexRedactor:
    """Mask common secret/PII patterns with ``[REDACTED:<kind>]`` tokens."""

    def redact(self, text: str) -> str:
        for kind, pattern in _PATTERNS:
            text = pattern.sub(f"[REDACTED:{kind}]", text)
        return text


def build_redactor(settings) -> Redactor:
    return RegexRedactor() if settings.redact_pii else NullRedactor()
