"""Optional OpenTelemetry tracing helpers.

The core pipeline emits spans through :func:`span`, which is a no-op unless
OpenTelemetry is installed *and* a tracer provider has been configured (see
``context_bridge.api.tracing.setup_tracing``). This keeps the domain layer free
of any hard tracing dependency — install the ``otel`` extra to light it up.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

try:  # pragma: no cover - exercised only when the otel extra is installed
    from opentelemetry import trace as _otel_trace

    _tracer = _otel_trace.get_tracer("context_bridge")
except Exception:  # pragma: no cover
    _tracer = None


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """Start a span if tracing is available, else act as a no-op."""
    if _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name) as current:  # pragma: no cover
        for key, value in attributes.items():
            current.set_attribute(key, value)
        yield current
