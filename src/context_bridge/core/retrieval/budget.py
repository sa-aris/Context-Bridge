"""Token-budgeted context assembly.

The whole point of Context Bridge is to hand an agent the *smallest* slice of
memory that answers its task. This assembler packs the highest-ranked chunks
into a single context string while strictly respecting a token budget, and
reports exactly how many tokens were spent.
"""

from __future__ import annotations

from context_bridge.core.models import AssembledContext, RetrievedChunk
from context_bridge.tokenizer import count_tokens, truncate_to_tokens

_MIN_TAIL_TOKENS = 32


def assemble(
    chunks: list[RetrievedChunk],
    *,
    token_budget: int,
    expand_parents: bool = False,
    separator: str = "\n\n---\n\n",
) -> AssembledContext:
    """Pack ``chunks`` into a context string within ``token_budget`` tokens.

    Chunks are consumed in the order given (already ranked). The final chunk may
    be truncated to fill the remaining budget when there is meaningful room left.
    """
    included: list[RetrievedChunk] = []
    parts: list[str] = []
    used = 0
    sep_tokens = count_tokens(separator)

    for chunk in chunks:
        text = chunk.parent_text if (expand_parents and chunk.parent_text) else chunk.content
        text = text.strip()
        if not text:
            continue
        cost = count_tokens(text) + (sep_tokens if parts else 0)
        if used + cost <= token_budget:
            parts.append(text)
            included.append(chunk)
            used += cost
            continue

        remaining = token_budget - used - (sep_tokens if parts else 0)
        if remaining >= _MIN_TAIL_TOKENS:
            truncated = truncate_to_tokens(text, remaining)
            if truncated:
                parts.append(truncated)
                included.append(chunk)
                used += count_tokens(truncated) + (sep_tokens if len(parts) > 1 else 0)
        break

    return AssembledContext(
        context=separator.join(parts),
        chunks=included,
        tokens_used=used,
    )
