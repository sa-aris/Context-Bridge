"""Provider-agnostic token counting.

Token budgets are the core cost-control lever of Context Bridge, so counting
needs to be cheap and dependency-tolerant. We use ``tiktoken`` when available
and fall back to a whitespace heuristic otherwise. The chosen encoding is a
reasonable proxy across modern tokenizers for budgeting purposes.
"""

from __future__ import annotations

from functools import lru_cache

try:  # pragma: no cover - exercised indirectly
    import tiktoken

    _HAS_TIKTOKEN = True
except Exception:  # pragma: no cover
    _HAS_TIKTOKEN = False


@lru_cache
def _encoding():  # pragma: no cover - thin wrapper around tiktoken
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Return an estimated token count for ``text``."""
    if not text:
        return 0
    if _HAS_TIKTOKEN:
        try:
            return len(_encoding().encode(text))
        except Exception:  # pragma: no cover - defensive
            pass
    # Heuristic fallback: ~4 chars/token, but never undercount short words.
    return max(len(text.split()), (len(text) + 3) // 4)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate ``text`` so that it fits within ``max_tokens``."""
    if max_tokens <= 0:
        return ""
    if count_tokens(text) <= max_tokens:
        return text
    if _HAS_TIKTOKEN:
        try:
            enc = _encoding()
            return enc.decode(enc.encode(text)[:max_tokens])
        except Exception:  # pragma: no cover - defensive
            pass
    # Fallback: trim by words.
    return " ".join(text.split()[:max_tokens])
