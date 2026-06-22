from __future__ import annotations

from context_bridge.tokenizer import count_tokens, truncate_to_tokens


def test_count_is_zero_for_empty():
    assert count_tokens("") == 0


def test_count_grows_with_length():
    short = count_tokens("hello world")
    longer = count_tokens("hello world " * 50)
    assert longer > short


def test_truncate_respects_limit():
    text = " ".join(["word"] * 100)
    truncated = truncate_to_tokens(text, 10)
    assert count_tokens(truncated) <= 10


def test_truncate_noop_when_already_small():
    assert truncate_to_tokens("tiny", 100) == "tiny"


def test_truncate_zero_is_empty():
    assert truncate_to_tokens("anything", 0) == ""
