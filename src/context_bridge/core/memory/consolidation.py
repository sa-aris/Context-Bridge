"""Reflective consolidation: cluster related memories and synthesize insights.

Run periodically (or on demand), this is the "sleep" phase of the shared brain:
it groups semantically-related memories in a namespace and writes back a single
higher-order summary per cluster. Over time the pool surfaces conclusions that no
single agent wrote, and redundant detail collapses into durable insight.
"""

from __future__ import annotations

import numpy as np


def cluster_by_similarity(vectors: list[list[float]], *, threshold: float) -> list[list[int]]:
    """Greedy single-link clustering of row indices by cosine similarity.

    Returns a list of clusters, each a list of indices into ``vectors``.
    """
    n = len(vectors)
    if n == 0:
        return []
    mat = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = mat / norms

    unassigned = set(range(n))
    clusters: list[list[int]] = []
    while unassigned:
        seed = unassigned.pop()
        members = [seed]
        sims = unit @ unit[seed]
        for j in list(unassigned):
            if sims[j] >= threshold:
                members.append(j)
                unassigned.discard(j)
        clusters.append(sorted(members))
    return clusters
