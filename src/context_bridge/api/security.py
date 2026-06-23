"""API-key authentication, namespace authorization and rate limiting.

All opt-in: with no API keys the service is open; with ``RATE_LIMIT_PER_MINUTE=0``
the limiter is a no-op. API keys may be scoped to specific namespaces for
multi-tenant deployments. The in-memory limiter suits a single replica; the
Redis backend shares state across replicas for horizontal scaling.
"""

from __future__ import annotations

import hmac
import threading
import time
from collections import defaultdict, deque
from typing import Protocol

from fastapi import HTTPException, Request, status

_API_KEY_HEADER = "x-api-key"
_WINDOW_SECONDS = 60


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #
class RateLimiter(Protocol):
    def allow(self, identity: str) -> bool: ...


class InMemoryRateLimiter:
    """Per-identity sliding-window limiter held in process memory."""

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


class RedisRateLimiter:
    """Fixed-window limiter backed by Redis (shared across replicas)."""

    def __init__(self, per_minute: int, url: str) -> None:
        import redis

        self.per_minute = per_minute
        self.client = redis.Redis.from_url(url, decode_responses=True)

    def allow(self, identity: str) -> bool:  # pragma: no cover - requires redis
        if self.per_minute <= 0:
            return True
        window = int(time.time()) // _WINDOW_SECONDS
        key = f"cb:rl:{identity}:{window}"
        pipe = self.client.pipeline()
        pipe.incr(key)
        pipe.expire(key, _WINDOW_SECONDS)
        count, _ = pipe.execute()
        return int(count) <= self.per_minute


def build_rate_limiter(settings) -> RateLimiter:
    """Construct the configured rate limiter."""
    if settings.rate_limit_backend.lower() == "redis":
        return RedisRateLimiter(settings.rate_limit_per_minute, settings.redis_url)
    return InMemoryRateLimiter(settings.rate_limit_per_minute)


# --------------------------------------------------------------------------- #
# Authentication & authorization
# --------------------------------------------------------------------------- #
def _identity(request: Request) -> str:
    api_key = request.headers.get(_API_KEY_HEADER)
    if api_key:
        return f"key:{api_key}"
    client = request.client
    return f"ip:{client.host if client else 'unknown'}"


def _matches_any(provided: str, keys: set[str]) -> bool:
    """Constant-time membership check to avoid leaking timing information."""
    return any(hmac.compare_digest(provided, key) for key in keys)


async def api_key_guard(request: Request) -> None:
    """Reject requests lacking a valid API key, when keys are configured.

    On success, stashes the caller's key on ``request.state`` so the RBAC layer
    can resolve its namespace/operation rules.
    """
    settings = request.app.state.settings
    keys = settings.api_key_set()
    if not keys:
        request.state.api_key = None
        return

    provided = request.headers.get(_API_KEY_HEADER)
    if not provided or not _matches_any(provided, keys):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or missing API key"
        )
    request.state.api_key = provided


async def rate_limit_guard(request: Request) -> None:
    """Reject requests that exceed the configured per-minute rate."""
    limiter: RateLimiter | None = getattr(request.app.state, "rate_limiter", None)
    if limiter is None:
        return
    if not limiter.allow(_identity(request)):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="rate limit exceeded"
        )


def authorize(request: Request, namespace: str, operation: str) -> None:
    """Ensure the authenticated caller may ``operation`` on ``namespace``."""
    access = request.app.state.access
    key = getattr(request.state, "api_key", None)
    if access.allows(key, namespace, operation):
        return
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=f"'{operation}' on namespace '{namespace}' is not permitted for this API key",
    )
