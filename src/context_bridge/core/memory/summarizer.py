"""Summarisation providers used by ``summarize-before-store`` and compaction.

The default is extractive (frequency-weighted sentence scoring): it needs no
model and never hallucinates, which matters when the output is written back
into shared memory. An LLM-backed abstractive summariser can be slotted in
behind the same protocol.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Protocol, runtime_checkable

logger = logging.getLogger("context_bridge")

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "then",
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
    "we",
    "you",
    "they",
    "i",
    "from",
    "so",
    "not",
}


@runtime_checkable
class Summarizer(Protocol):
    def summarize(self, text: str, *, max_sentences: int = 5) -> str: ...


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


_LLM_PROMPT = (
    "Summarize the following notes into at most {n} concise sentences. "
    "Preserve concrete facts, decisions, identifiers and numbers. "
    "Do not invent information not present in the notes.\n\n"
    "NOTES:\n{text}\n\nSUMMARY:"
)


class LLMSummarizer:
    """Abstractive summariser via any OpenAI-compatible chat endpoint.

    Stays vendor-neutral (configurable base URL / model) and degrades to the
    extractive summariser on any error, so a flaky or absent LLM never breaks
    the write path.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = 30.0,
        max_tokens: int = 512,
        fallback: Summarizer | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens
        self._fallback = fallback or ExtractiveSummarizer()

    def summarize(self, text: str, *, max_sentences: int = 5) -> str:
        text = text.strip()
        if not text:
            return ""
        try:
            return self._call_llm(text, max_sentences)
        except Exception as exc:  # pragma: no cover - network/endpoint dependent
            logger.warning("llm summariser failed, falling back to extractive: %s", exc)
            return self._fallback.summarize(text, max_sentences=max_sentences)

    def _call_llm(self, text: str, max_sentences: int) -> str:
        import httpx

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": _LLM_PROMPT.format(n=max_sentences, text=text)}
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0.2,
        }
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return content.strip()


def build_summarizer(settings) -> Summarizer:
    """Construct the configured summariser."""
    provider = settings.summarizer_provider.lower()
    if provider == "extractive":
        return ExtractiveSummarizer()
    if provider == "llm":
        return LLMSummarizer(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout,
            max_tokens=settings.llm_max_tokens,
        )
    raise ValueError(f"Unknown summarizer_provider: {settings.summarizer_provider!r}")
