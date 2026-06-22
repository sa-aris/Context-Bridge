"""A thin, framework-agnostic client for the Context Bridge API.

Drop this into any agent (AutoGen, CrewAI, a bare loop, ...) so it can write
its outputs to shared memory and recall a task-scoped, budget-bounded slice
instead of passing whole transcripts around.
"""

from __future__ import annotations

from types import TracebackType
from typing import Any

import httpx

_DEFAULT_TIMEOUT = 30.0


class ContextBridgeClient:
    """Synchronous HTTP client mirroring the server's memory API."""

    def __init__(
        self, base_url: str = "http://localhost:8000", *, timeout: float = _DEFAULT_TIMEOUT
    ):
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    # -- lifecycle --------------------------------------------------------
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

    # -- memory -----------------------------------------------------------
    def remember(
        self,
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
        """Write a memory to the shared pool."""
        payload = {
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
        return self._post("/memory/write", payload)

    def recall(
        self,
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
        """Recall a task-scoped, budget-bounded context slice."""
        payload = {
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
        return self._post("/memory/query", payload)

    def summarize(
        self,
        session_id: str,
        *,
        namespace: str = "default",
        max_sentences: int = 5,
        store_summary: bool = True,
    ) -> dict[str, Any]:
        payload = {
            "session_id": session_id,
            "namespace": namespace,
            "max_sentences": max_sentences,
            "store_summary": store_summary,
        }
        return self._post("/memory/summarize", payload)

    def timeline(self, session_id: str, *, limit: int = 100) -> dict[str, Any]:
        resp = self._client.get(f"/sessions/{session_id}/timeline", params={"limit": limit})
        resp.raise_for_status()
        return resp.json()

    def get(self, record_id: str) -> dict[str, Any]:
        resp = self._client.get(f"/memory/{record_id}")
        resp.raise_for_status()
        return resp.json()

    def delete(self, record_id: str) -> None:
        resp = self._client.delete(f"/memory/{record_id}")
        resp.raise_for_status()

    # -- internals --------------------------------------------------------
    def _post(self, path: str, payload: dict) -> dict[str, Any]:
        resp = self._client.post(path, json=payload)
        resp.raise_for_status()
        return resp.json()
