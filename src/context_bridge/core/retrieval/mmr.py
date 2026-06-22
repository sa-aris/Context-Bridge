"""Maximal Marginal Relevance (MMR) selection.

MMR trades off relevance against novelty so the returned set is not three
paraphrases of the same fact. This directly attacks redundancy, which is the
biggest hidden token cost in naive top-k retrieval.
"""

from __future__ import annotations

import numpy as np

from context_bridge.core.models import RetrievedChunk


def _normalise_relevance(chunks: list[RetrievedChunk]) -> dict[str, float]:
    scores = [c.score for c in chunks]
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return {c.id: 1.0 for c in chunks}
    return {c.id: (c.score - lo) / (hi - lo) for c in chunks}


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def mmr_select(
    chunks: list[RetrievedChunk],
    *,
    lambda_: float = 0.6,
    top_k: int = 8,
) -> list[RetrievedChunk]:
    """Re-order ``chunks`` by MMR and return the best ``top_k``."""
    if not chunks:
        return []
    if len(chunks) <= top_k:
        return chunks

    relevance = _normalise_relevance(chunks)
    vectors = {
        c.id: np.asarray(c.dense, dtype=np.float32) if c.dense is not None else None
        for c in chunks
    }

    selected: list[RetrievedChunk] = []
    remaining = list(chunks)
    while remaining and len(selected) < top_k:
        best: RetrievedChunk | None = None
        best_score = float("-inf")
        for cand in remaining:
            diversity = 0.0
            cand_vec = vectors[cand.id]
            if selected and cand_vec is not None:
                sims = [
                    _cosine(cand_vec, sel_vec)
                    for s in selected
                    if (sel_vec := vectors[s.id]) is not None
                ]
                diversity = max(sims) if sims else 0.0
            value = lambda_ * relevance[cand.id] - (1.0 - lambda_) * diversity
            if value > best_score:
                best_score = value
                best = cand
        assert best is not None
        selected.append(best)
        remaining.remove(best)
    return selected
