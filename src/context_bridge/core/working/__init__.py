"""Short-term working memory providers (per-session scratchpad)."""

from __future__ import annotations

from context_bridge.config import Settings
from context_bridge.core.working.base import WorkingMemory
from context_bridge.core.working.memory_store import InMemoryWorkingStore


def build_working_memory(settings: Settings) -> WorkingMemory:
    """Construct the configured working-memory store."""
    provider = settings.working_provider.lower()
    if provider == "memory":
        return InMemoryWorkingStore(ttl_seconds=settings.working_ttl_seconds)
    if provider == "redis":
        from context_bridge.core.working.redis_store import RedisWorkingStore

        return RedisWorkingStore(url=settings.redis_url, ttl_seconds=settings.working_ttl_seconds)
    raise ValueError(f"Unknown working_provider: {settings.working_provider!r}")


__all__ = ["WorkingMemory", "InMemoryWorkingStore", "build_working_memory"]
