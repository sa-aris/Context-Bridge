from __future__ import annotations

from context_bridge.config import Settings
from context_bridge.core.memory.summarizer import (
    ExtractiveSummarizer,
    LLMSummarizer,
    build_summarizer,
)

_TEXT = (
    "The migration script must run before the API deploy. "
    "Table orders is referenced by three new endpoints. "
    "The rollback plan restores the previous schema snapshot. "
    "Deployment is scheduled for Friday evening. "
    "The on-call engineer is responsible for monitoring error rates."
)


def test_extractive_reduces_sentence_count():
    summary = ExtractiveSummarizer().summarize(_TEXT, max_sentences=2)
    assert summary
    assert summary.count(".") <= 2


def test_extractive_handles_empty():
    assert ExtractiveSummarizer().summarize("") == ""


def test_build_summarizer_default_is_extractive():
    summarizer = build_summarizer(Settings())
    assert isinstance(summarizer, ExtractiveSummarizer)


def test_llm_summarizer_falls_back_when_endpoint_unreachable():
    # Unroutable endpoint -> the call fails and we expect the extractive fallback.
    summarizer = LLMSummarizer(
        base_url="http://127.0.0.1:1/v1", model="none", timeout=0.2
    )
    summary = summarizer.summarize(_TEXT, max_sentences=2)
    assert summary  # non-empty: came from the fallback
