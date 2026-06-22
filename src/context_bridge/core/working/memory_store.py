"""In-process working memory (default; no external services required)."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class InMemoryWorkingStore:
    """Thread-safe, TTL'd per-session ring buffer kept in process memory."""

    def __init__(self, ttl_seconds: int = 3600, max_items: int = 200) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_items = max_items
        self._data: dict[str, deque[tuple[float, dict]]] = defaultdict(
            lambda: deque(maxlen=max_items)
        )
        self._lock = threading.Lock()

    def append(self, session_id: str, item: dict) -> None:
        with self._lock:
            self._data[session_id].append((time.time(), item))

    def recent(self, session_id: str, limit: int = 20) -> list[dict]:
        cutoff = time.time() - self.ttl_seconds
        with self._lock:
            entries = [item for ts, item in self._data.get(session_id, ()) if ts >= cutoff]
        return entries[-limit:]

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._data.pop(session_id, None)
