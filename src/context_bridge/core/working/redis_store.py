"""Redis-backed working memory for multi-process / multi-replica deployments."""

from __future__ import annotations

import json

_KEY = "cb:working:{session_id}"


class RedisWorkingStore:
    """Stores each session's scratchpad as a capped, expiring Redis list."""

    def __init__(self, url: str, ttl_seconds: int = 3600, max_items: int = 200) -> None:
        import redis

        self.client = redis.Redis.from_url(url, decode_responses=True)
        self.ttl_seconds = ttl_seconds
        self.max_items = max_items

    def append(self, session_id: str, item: dict) -> None:
        key = _KEY.format(session_id=session_id)
        pipe = self.client.pipeline()
        pipe.rpush(key, json.dumps(item))
        pipe.ltrim(key, -self.max_items, -1)
        pipe.expire(key, self.ttl_seconds)
        pipe.execute()

    def recent(self, session_id: str, limit: int = 20) -> list[dict]:
        key = _KEY.format(session_id=session_id)
        raw = self.client.lrange(key, -limit, -1)
        return [json.loads(r) for r in raw]

    def clear(self, session_id: str) -> None:
        self.client.delete(_KEY.format(session_id=session_id))
