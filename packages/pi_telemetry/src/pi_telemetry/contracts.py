"""Vendor-neutral telemetry contracts for Pi (Python port of @earendil-works/pi telemetry).

These are the runtime-relevant pieces of the upstream TypeScript contract surface.
The upstream package is heavily type-level (compile-time span-schema inference); this
port keeps the same *runtime* behavior and exposes the schema definitions as plain data
so they remain inspectable, while explicit validation is intentionally left to callers
(upstream comment: "Schema values are used only for type inference; no runtime schema
validation is performed.").
"""

from __future__ import annotations

from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)

# ---------------------------------------------------------------------------
# Primitive value types
# ---------------------------------------------------------------------------

# A single attribute value. Strings, numbers, booleans, or homogeneous sequences
# of those. Mirrors the upstream `AttributeValue` union.
AttributeValue = Union[
    str,
    int,
    float,
    bool,
    Sequence[str],
    Sequence[int],
    Sequence[float],
    Sequence[bool],
]

# Attributes may explicitly carry `None` to signal "drop this key" (upstream treats
# `undefined` as "do not record"). We model that as a Python `None` value.
SpanAttributes = Dict[str, Optional[AttributeValue]]


class SpanOptions(Dict[str, Any]):
    """``{"name": str, "attributes"?: SpanAttributes}``."""


class SpanStatusOk(Dict[str, Any]):
    """``{"status": "ok"}``."""


class SpanStatusError(Dict[str, Any]):
    """``{"status": "error", "error"?: {"name": str, "message": str}}``."""


SpanStatus = Union[SpanStatusOk, SpanStatusError]


# A callback receives a span and may return a value or an awaitable.
CallbackResult = Union["T", Awaitable["T"]]
SpanCallback = Callable[["TelemetrySpan"], Any]


@runtime_checkable
class TelemetrySpan(Protocol):
    """A span handle handed to a callback. Also a context for child spans."""

    def start_span(self, options: SpanOptions, callback: SpanCallback) -> Awaitable[Any]:
        """Open a child span and run ``callback`` with the child span handle."""
        ...

    def add_event(self, name: str, attributes: Optional[SpanAttributes] = None) -> None:
        """Record a named event with optional attributes on this span."""
        ...

    def set_attributes(self, attributes: SpanAttributes) -> None:
        """Merge attributes into this span (later keys win; ``None`` drops a key)."""
        ...

    def set_status(self, status: SpanStatus) -> None:
        """Set the span status explicitly (``ok`` or ``error``)."""
        ...


@runtime_checkable
class TelemetryContext(Protocol):
    """The entry point used by applications to record telemetry."""

    def start_span(self, options: SpanOptions, callback: SpanCallback) -> Awaitable[Any]:
        """Open a span, run ``callback`` with the span handle, return its result."""
        ...


__all__ = [
    "AttributeValue",
    "SpanAttributes",
    "SpanOptions",
    "SpanStatus",
    "SpanStatusOk",
    "SpanStatusError",
    "TelemetrySpan",
    "TelemetryContext",
]
