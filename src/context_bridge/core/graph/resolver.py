"""Entity-name normalization for ontology alignment.

Different agents name the same thing differently ("db one", "Database-1",
"database  one"). :func:`normalize` collapses surface variants to a comparison
key so they can be merged onto a single canonical entity.
"""

from __future__ import annotations

import re

_SEP_RE = re.compile(r"[\s_\-./]+")


def normalize(name: str) -> str:
    """Return a comparison key: lowercased, separator- and space-collapsed."""
    return _SEP_RE.sub(" ", name.strip().lower()).strip()


def choose_canonical(variants: list[str], edge_counts: dict[str, int]) -> str:
    """Pick the most-established variant: most edges, then shortest, then a-z."""
    return sorted(variants, key=lambda v: (-edge_counts.get(v, 0), len(v), v))[0]
