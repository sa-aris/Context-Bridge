"""The webhook event emitter and its no-op default."""

from __future__ import annotations

from context_bridge.core.events import NullEmitter, WebhookEmitter, build_emitter


def test_build_emitter_is_null_without_urls():
    assert isinstance(build_emitter([]), NullEmitter)


def test_build_emitter_uses_webhooks_when_configured():
    assert isinstance(build_emitter(["https://example.test/hook"]), WebhookEmitter)


def test_null_emitter_is_silent():
    # Must accept any event without raising or doing anything.
    assert NullEmitter().emit("conflict.opened", "ns", {"x": 1}) is None
