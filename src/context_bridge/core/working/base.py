"""The working-memory contract."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class WorkingMemory(Protocol):
    """A fast, ephemeral, per-session scratchpad.

    Holds the most recent turns/notes so an agent has immediate continuity
    without paying to re-embed and re-retrieve them. Entries expire via TTL.
    """

    def append(self, session_id: str, item: dict) -> None:
        """Record an item against ``session_id``."""
        ...

    def recent(self, session_id: str, limit: int = 20) -> list[dict]:
        """Return up to ``limit`` most-recent, non-expired items (oldest first)."""
        ...

    def clear(self, session_id: str) -> None:
        """Drop all items for ``session_id``."""
        ...
