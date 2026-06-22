from __future__ import annotations

from context_bridge.api.tracing import setup_tracing
from context_bridge.config import Settings
from context_bridge.core.tracing import span


def test_span_is_a_noop_context_manager_without_otel():
    # Without the otel extra / a configured provider, span must not raise.
    with span("unit.test", foo="bar") as s:
        assert s is None


def test_setup_tracing_noop_when_disabled():
    class _DummyApp:
        pass

    # Should return cleanly without touching the app when tracing is disabled.
    setup_tracing(_DummyApp(), Settings(tracing_enabled=False))


def test_setup_tracing_safe_when_extra_missing():
    class _DummyApp:
        pass

    # Enabled but the optional dependency is absent -> warn and continue.
    setup_tracing(_DummyApp(), Settings(tracing_enabled=True))
