"""Outbound event notifications (webhooks) for notable memory events.

The shared pool can tell external systems when something noteworthy happens —
a contradiction is opened, a lesson is captured, a conflict is auto-resolved —
so dashboards, chat-ops or downstream automations can react. Delivery is
best-effort and never blocks or fails a memory operation.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import httpx

from context_bridge.core.models import now_ts

logger = logging.getLogger("context_bridge.events")


@runtime_checkable
class EventEmitter(Protocol):
    """Emits a named event for a namespace with an arbitrary payload."""

    def emit(self, event_type: str, namespace: str, data: dict) -> None: ...


class NullEmitter:
    """The default no-op emitter (events disabled)."""

    def emit(self, event_type: str, namespace: str, data: dict) -> None:  # noqa: D102
        return None


class WebhookEmitter:
    """Posts each event to one or more webhook URLs, best-effort."""

    def __init__(self, urls: list[str], *, timeout: float = 5.0) -> None:
        self.urls = urls
        self.timeout = timeout

    def emit(self, event_type: str, namespace: str, data: dict) -> None:
        payload = {
            "type": event_type,
            "namespace": namespace,
            "data": data,
            "ts": now_ts(),
        }
        for url in self.urls:
            try:
                httpx.post(url, json=payload, timeout=self.timeout)
            except Exception:  # pragma: no cover - delivery must never break a write
                logger.warning("webhook delivery to %s failed for %s", url, event_type)


class CompositeEmitter:
    """Fans an event out to several emitters; one failing never blocks the rest."""

    def __init__(self, emitters: list[EventEmitter]) -> None:
        self.emitters = emitters

    def emit(self, event_type: str, namespace: str, data: dict) -> None:
        for emitter in self.emitters:
            try:
                emitter.emit(event_type, namespace, data)
            except Exception:  # pragma: no cover - one sink must not break another
                logger.warning("event sink %r failed for %s", emitter, event_type)


def build_emitter(urls: list[str], *, timeout: float = 5.0) -> EventEmitter:
    """Return a webhook emitter when URLs are configured, else a no-op."""
    return WebhookEmitter(urls, timeout=timeout) if urls else NullEmitter()
