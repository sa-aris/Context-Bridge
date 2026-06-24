"""Extract (subject, relation, object) triples to build a knowledge graph.

The default :class:`RuleBasedExtractor` is deterministic and dependency-free: it
matches a curated set of relation phrases and captures the entity on each side.
It favours precision over recall — clean signal beats noisy edges in a shared
graph. Plug an LLM-backed extractor behind :class:`Extractor` for full coverage.
"""

from __future__ import annotations

import re
from typing import NamedTuple, Protocol, runtime_checkable


class Triple(NamedTuple):
    source: str
    relation: str
    target: str


@runtime_checkable
class Extractor(Protocol):
    def extract(self, text: str) -> list[Triple]: ...


# Relation phrases, longest first so "is part of" wins over "is".
_RELATIONS = [
    "is part of",
    "depends on",
    "connects to",
    "reads from",
    "writes to",
    "belongs to",
    "relates to",
    "talks to",
    "requires",
    "contains",
    "manages",
    "produces",
    "consumes",
    "owns",
    "uses",
    "calls",
]
_ENTITY = r"[A-Za-z][A-Za-z0-9_./-]*(?:\s+[A-Za-z0-9_./-]+){0,3}"
_SENTENCE_RE = re.compile(r"[.!?\n]")
_PATTERNS = [
    (rel, re.compile(rf"({_ENTITY})\s+{re.escape(rel)}\s+({_ENTITY})", re.IGNORECASE))
    for rel in _RELATIONS
]
_STOP_PREFIXES = ("the ", "a ", "an ", "it ", "its ", "their ", "this ", "that ")


def _clean(entity: str) -> str:
    entity = entity.strip().strip(".,;:\"'()").strip()
    low = entity.lower()
    for prefix in _STOP_PREFIXES:
        if low.startswith(prefix):
            entity = entity[len(prefix) :].strip()
            break
    return entity


class RuleBasedExtractor:
    """Pattern-based triple extraction (see module docstring)."""

    def extract(self, text: str) -> list[Triple]:
        triples: list[Triple] = []
        seen: set[Triple] = set()
        for sentence in _SENTENCE_RE.split(text):
            for relation, pattern in _PATTERNS:
                for match in pattern.finditer(sentence):
                    source = _clean(match.group(1))
                    target = _clean(match.group(2))
                    if not source or not target or source.lower() == target.lower():
                        continue
                    triple = Triple(source=source, relation=relation, target=target)
                    if triple not in seen:
                        seen.add(triple)
                        triples.append(triple)
        return triples


def build_extractor(settings) -> Extractor:
    """Construct the configured extractor (rule-based for now)."""
    return RuleBasedExtractor()
