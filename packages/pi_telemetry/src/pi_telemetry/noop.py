"""No-op telemetry context used when an application provides none.

The no-op context still *runs* every callback exactly once and returns its result
(or propagates its rejection) so that application code behaves identically whether or
not telemetry is wired up. It simply records nothing.
"""

from __future__ import annotations

import inspect
from typing import Any, Awaitable

from .contracts import SpanOptions, TelemetryContext, TelemetrySpan


async def _invoke(callback, span):
    result = callback(span)
    if inspect.isawaitable(result):
        return await result
    return result


class _NoopSpan:
    """A span that records nothing but still executes child callbacks."""

    def start_span(self, options: SpanOptions, callback) -> Awaitable[Any]:
        return _start_noop_span(options, callback)

    def add_event(self, name: str, attributes=None) -> None:
        return None

    def set_attributes(self, attributes) -> None:
        return None

    def set_status(self, status) -> None:
        return None


_noop_span = _NoopSpan()


async def _start_noop_span(options: SpanOptions, callback) -> Any:
    # Upstream wraps callback execution so a synchronous throw becomes a rejected
    # promise with the *same* error identity. In Python a synchronous raise inside an
    # async function already propagates with the same traceback/identity, so we just
    # invoke and let exceptions propagate naturally.
    return await _invoke(callback, _noop_span)


class _NoopTelemetryContext:
    """Implements both :class:`TelemetryContext` and :class:`TelemetrySpan`."""

    def start_span(self, options: SpanOptions, callback) -> Awaitable[Any]:
        return _start_noop_span(options, callback)

    def add_event(self, name: str, attributes=None) -> None:
        return None

    def set_attributes(self, attributes) -> None:
        return None

    def set_status(self, status) -> None:
        return None


#: Shared telemetry context used when an application does not provide one.
NOOP_TELEMETRY_CONTEXT: TelemetryContext = _NoopTelemetryContext()
