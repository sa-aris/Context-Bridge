"""Thin, framework-agnostic clients for the Context Bridge API.

Drop these into any agent (AutoGen, CrewAI, a bare loop, ...) so it can write
its outputs to shared memory and recall a task-scoped, budget-bounded slice
instead of passing whole transcripts around. Both a synchronous and an
asynchronous client are provided.
"""

from __future__ import annotations

from types import TracebackType
from typing import Any

import httpx

_DEFAULT_TIMEOUT = 30.0
_API_PREFIX = "/v1"


def _write_payload(
    content: str,
    *,
    agent_id: str,
    session_id: str,
    task_id: str | None = None,
    namespace: str = "default",
    tags: list[str] | None = None,
    confidence: float = 1.0,
    metadata: dict | None = None,
    source: str | None = None,
    ttl_seconds: int | None = None,
    dedup: bool = True,
    summarize_before_store: bool = False,
) -> dict[str, Any]:
    return {
        "content": content,
        "agent_id": agent_id,
        "session_id": session_id,
        "task_id": task_id,
        "namespace": namespace,
        "tags": tags or [],
        "confidence": confidence,
        "metadata": metadata or {},
        "source": source,
        "ttl_seconds": ttl_seconds,
        "dedup": dedup,
        "summarize_before_store": summarize_before_store,
    }


def _recall_payload(
    query: str,
    *,
    namespace: str = "default",
    agent_id: str = "system",
    session_id: str | None = None,
    top_k: int | None = None,
    token_budget: int | None = None,
    filters: dict | None = None,
    rerank: bool = True,
    expand_parents: bool = False,
) -> dict[str, Any]:
    return {
        "query": query,
        "namespace": namespace,
        "agent_id": agent_id,
        "session_id": session_id,
        "top_k": top_k,
        "token_budget": token_budget,
        "filters": filters,
        "rerank": rerank,
        "expand_parents": expand_parents,
    }


def _headers(api_key: str | None) -> dict[str, str]:
    return {"X-API-Key": api_key} if api_key else {}


class ContextBridgeClient:
    """Synchronous HTTP client mirroring the server's memory API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        *,
        api_key: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ):
        self._client = httpx.Client(
            base_url=base_url.rstrip("/") + _API_PREFIX,
            timeout=timeout,
            headers=_headers(api_key),
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> ContextBridgeClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def remember(self, content: str, **kwargs: Any) -> dict[str, Any]:
        """Write a memory to the shared pool."""
        return self._json("POST", "/memory/write", _write_payload(content, **kwargs))

    def remember_batch(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        """Write many memories in a single request."""
        payload = {"items": [_write_payload(i.pop("content"), **i) for i in items]}
        return self._json("POST", "/memory/write_batch", payload)

    def recall(self, query: str, **kwargs: Any) -> dict[str, Any]:
        """Recall a task-scoped, budget-bounded context slice."""
        return self._json("POST", "/memory/query", _recall_payload(query, **kwargs))

    def list(
        self, *, namespace: str = "default", limit: int = 50, cursor: str | None = None
    ) -> dict[str, Any]:
        params = {"namespace": namespace, "limit": limit, "cursor": cursor}
        return self._json(
            "GET", "/memory", params={k: v for k, v in params.items() if v is not None}
        )

    def summarize(self, session_id: str, **kwargs: Any) -> dict[str, Any]:
        return self._json("POST", "/memory/summarize", {"session_id": session_id, **kwargs})

    def timeline(self, session_id: str, *, limit: int = 100) -> dict[str, Any]:
        return self._json("GET", f"/sessions/{session_id}/timeline", params={"limit": limit})

    def get(self, record_id: str) -> dict[str, Any]:
        return self._json("GET", f"/memory/{record_id}")

    def delete(self, record_id: str) -> None:
        resp = self._client.delete(f"/memory/{record_id}")
        resp.raise_for_status()

    def _json(
        self,
        method: str,
        path: str,
        json: dict | None = None,
        params: dict | None = None,
    ) -> dict[str, Any]:
        resp = self._client.request(method, path, json=json, params=params)
        resp.raise_for_status()
        return resp.json()


class AsyncContextBridgeClient:
    """Asynchronous counterpart of :class:`ContextBridgeClient`."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        *,
        api_key: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ):
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + _API_PREFIX,
            timeout=timeout,
            headers=_headers(api_key),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> AsyncContextBridgeClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def remember(self, content: str, **kwargs: Any) -> dict[str, Any]:
        return await self._json("POST", "/memory/write", _write_payload(content, **kwargs))

    async def recall(self, query: str, **kwargs: Any) -> dict[str, Any]:
        return await self._json("POST", "/memory/query", _recall_payload(query, **kwargs))

    async def get(self, record_id: str) -> dict[str, Any]:
        return await self._json("GET", f"/memory/{record_id}")

    async def _json(
        self,
        method: str,
        path: str,
        json: dict | None = None,
        params: dict | None = None,
    ) -> dict[str, Any]:
        resp = await self._client.request(method, path, json=json, params=params)
        resp.raise_for_status()
        return resp.json()
