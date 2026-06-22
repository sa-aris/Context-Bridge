"""Write governance: the rules that keep the shared pool healthy.

Left ungoverned, a shared memory pool degrades: agents re-write the same fact,
low-confidence guesses get retrieved and amplified, and stale entries linger.
:class:`WritePolicy` enforces confidence gating and near-duplicate suppression
so only novel, trustworthy memories are persisted.
"""

from __future__ import annotations

import numpy as np

from context_bridge.core.models import RetrievedChunk


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two dense vectors."""
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


class WritePolicy:
    """Decides whether a candidate chunk should be persisted."""

    def __init__(self, *, dedup_threshold: float = 0.95, min_confidence: float = 0.0) -> None:
        self.dedup_threshold = dedup_threshold
        self.min_confidence = min_confidence

    def passes_confidence(self, confidence: float) -> bool:
        return confidence >= self.min_confidence

    def is_duplicate(self, candidate_dense: list[float], neighbor: RetrievedChunk | None) -> bool:
        """True if ``candidate_dense`` is near-identical to an existing neighbor."""
        if neighbor is None or neighbor.dense is None:
            return False
        return cosine(candidate_dense, neighbor.dense) >= self.dedup_threshold
