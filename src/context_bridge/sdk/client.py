"""Thin, framework-agnostic clients for the Context Bridge API.

Drop these into any agent (AutoGen, CrewAI, a bare loop, ...) so it can write
its outputs to shared memory and recall a task-scoped, budget-bounded slice
instead of passing whole transcripts around. Both a synchronous and an
asynchronous client are provided.
"""

from __future__ import annotations

from collections.abc import Sequence
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

    def forget(
        self, *, namespace: str | None = None, session_id: str | None = None
    ) -> dict[str, Any]:
        """Erase all memory for a namespace and/or session (right-to-be-forgotten)."""
        params = {"namespace": namespace, "session_id": session_id}
        return self._json("DELETE", "/memory", params={k: v for k, v in params.items() if v})

    # -- learning loop ----------------------------------------------------
    def feedback(
        self, memory_id: str, *, namespace: str = "default", useful: bool, weight: float = 1.0
    ) -> None:
        """Signal whether a recalled memory was useful, re-ranking future recall."""
        body = {"memory_id": memory_id, "namespace": namespace, "useful": useful, "weight": weight}
        resp = self._client.post("/memory/feedback", json=body)
        resp.raise_for_status()

    def record_outcome(self, session_id: str, *, success: bool, **kwargs: Any) -> dict[str, Any]:
        """Credit a session's memories and agents by its outcome (with optional lesson)."""
        return self._json(
            "POST", "/outcomes", {"session_id": session_id, "success": success, **kwargs}
        )

    def agents(self, *, namespace: str = "default", limit: int = 20) -> dict[str, Any]:
        return self._json("GET", "/agents", params={"namespace": namespace, "limit": limit})

    def create_procedure(self, title: str, steps: Sequence[str], **kwargs: Any) -> dict[str, Any]:
        return self._json("POST", "/procedures", {"title": title, "steps": list(steps), **kwargs})

    def procedures(self, *, namespace: str = "default", query: str | None = None) -> dict[str, Any]:
        params = {"namespace": namespace, "query": query}
        return self._json("GET", "/procedures", params={k: v for k, v in params.items() if v})

    # -- failure memory ---------------------------------------------------
    def record_lesson(self, trigger: str, guidance: str, **kwargs: Any) -> dict[str, Any]:
        """Capture a lesson so a past mistake is flagged before it recurs."""
        return self._json("POST", "/lessons", {"trigger": trigger, "guidance": guidance, **kwargs})

    def lessons(self, *, namespace: str = "default", limit: int = 100) -> dict[str, Any]:
        return self._json("GET", "/lessons", params={"namespace": namespace, "limit": limit})

    def confirm_lesson(self, lesson_id: str, *, namespace: str = "default") -> None:
        resp = self._client.post(f"/lessons/{lesson_id}/confirm", params={"namespace": namespace})
        resp.raise_for_status()

    def distill_lessons(self, *, namespace: str = "default") -> dict[str, Any]:
        return self._json("POST", "/lessons/distill", params={"namespace": namespace})

    def preflight(self, task: str, *, namespace: str = "default", limit: int = 5) -> dict[str, Any]:
        """Brief: lessons to avoid + playbooks that worked, before starting a task."""
        return self._json(
            "POST", "/preflight", {"task": task, "namespace": namespace, "limit": limit}
        )

    # -- truth maintenance & graph ---------------------------------------
    def conflicts(self, *, namespace: str = "default", status: str | None = None) -> dict[str, Any]:
        params = {"namespace": namespace, "status": status}
        return self._json("GET", "/conflicts", params={k: v for k, v in params.items() if v})

    def resolve_conflict(
        self, conflict_id: str, *, namespace: str = "default", winner_id: str | None = None
    ) -> None:
        resp = self._client.post(
            f"/conflicts/{conflict_id}/resolve",
            json={"winner_id": winner_id},
            params={"namespace": namespace},
        )
        resp.raise_for_status()

    def auto_resolve_conflicts(self, *, namespace: str = "default") -> dict[str, Any]:
        return self._json("POST", "/conflicts/auto-resolve", params={"namespace": namespace})

    def graph_neighbors(
        self, entity: str, *, namespace: str = "default", hops: int = 1
    ) -> dict[str, Any]:
        params = {"entity": entity, "namespace": namespace, "hops": hops}
        return self._json("GET", "/graph/neighbors", params=params)

    def align_graph(self, *, namespace: str = "default") -> dict[str, Any]:
        return self._json("POST", "/graph/align", {"namespace": namespace})

    def add_alias(
        self, alias: str, canonical: str, *, namespace: str = "default"
    ) -> dict[str, Any]:
        body = {"alias": alias, "canonical": canonical, "namespace": namespace}
        return self._json("POST", "/graph/aliases", body)

    # -- insight & operations --------------------------------------------
    def quality(self, *, namespace: str = "default") -> dict[str, Any]:
        return self._json("GET", "/quality", params={"namespace": namespace})

    def health(self, *, namespace: str = "default") -> dict[str, Any]:
        """The namespace memory-health panel."""
        return self._json("GET", f"/namespaces/{namespace}/health")

    def beliefs(self, query: str, *, namespace: str = "default", limit: int = 50) -> dict[str, Any]:
        """The belief timeline (memory diff) for a topic."""
        return self._json(
            "GET", f"/namespaces/{namespace}/beliefs", params={"query": query, "limit": limit}
        )

    def export_namespace(self, *, namespace: str = "default") -> dict[str, Any]:
        return self._json("GET", f"/namespaces/{namespace}/export")

    def import_namespace(self, payload: dict, *, namespace: str = "default") -> dict[str, Any]:
        return self._json("POST", f"/namespaces/{namespace}/import", payload)

    def run_maintenance(self) -> dict[str, Any]:
        return self._json("POST", "/maintenance/run")

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

    async def feedback(
        self, memory_id: str, *, namespace: str = "default", useful: bool, weight: float = 1.0
    ) -> None:
        body = {"memory_id": memory_id, "namespace": namespace, "useful": useful, "weight": weight}
        resp = await self._client.post("/memory/feedback", json=body)
        resp.raise_for_status()

    async def record_outcome(
        self, session_id: str, *, success: bool, **kwargs: Any
    ) -> dict[str, Any]:
        return await self._json(
            "POST", "/outcomes", {"session_id": session_id, "success": success, **kwargs}
        )

    async def preflight(
        self, task: str, *, namespace: str = "default", limit: int = 5
    ) -> dict[str, Any]:
        return await self._json(
            "POST", "/preflight", {"task": task, "namespace": namespace, "limit": limit}
        )

    async def quality(self, *, namespace: str = "default") -> dict[str, Any]:
        return await self._json("GET", "/quality", params={"namespace": namespace})

    async def health(self, *, namespace: str = "default") -> dict[str, Any]:
        return await self._json("GET", f"/namespaces/{namespace}/health")

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
