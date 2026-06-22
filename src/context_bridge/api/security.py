"""API-key authentication and in-process rate limiting.

Both are opt-in: with no API keys configured the service is open, and with
``RATE_LIMIT_PER_MINUTE=0`` the limiter is a no-op. The limiter is a simple
per-identity sliding window kept in process memory — adequate for a single
replica; front it with a shared store (e.g. Redis) for horizontal scaling.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request, status

_API_KEY_HEADER = "x-api-key"
_WINDOW_SECONDS = 60.0


class RateLimiter:
    """Per-identity sliding-window request limiter."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, identity: str) -> bool:
        if self.per_minute <= 0:
            return True
        now = time.time()
        cutoff = now - _WINDOW_SECONDS
        with self._lock:
            hits = self._hits[identity]
            while hits and hits[0] < cutoff:
                hits.popleft()
            if len(hits) >= self.per_minute:
                return False
            hits.append(now)
            return True


def _identity(request: Request) -> str:
    api_key = request.headers.get(_API_KEY_HEADER)
    if api_key:
        return f"key:{api_key}"
    client = request.client
    return f"ip:{client.host if client else 'unknown'}"


async def api_key_guard(request: Request) -> None:
    """Reject requests lacking a valid API key, when keys are configured."""
    keys = request.app.state.settings.api_key_set()
    if not keys:
        return
    provided = request.headers.get(_API_KEY_HEADER)
    if provided not in keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or missing API key"
        )


async def rate_limit_guard(request: Request) -> None:
    """Reject requests that exceed the configured per-minute rate."""
    limiter: RateLimiter | None = getattr(request.app.state, "rate_limiter", None)
    if limiter is None:
        return
    if not limiter.allow(_identity(request)):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="rate limit exceeded"
        )
