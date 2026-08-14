"""In-process recording telemetry context (reference implementation).

Mirrors upstream ``memory.ts``: spans are recorded in process memory with parent
linkage, an ``endSequence`` assigned at settle time, and fully passive recording
(malformed/unreadable telemetry payloads are silently ignored rather than crashing
the application). A span that has already settled becomes inert for mutation, but
still forwards child ``start_span`` calls to the no-op context so the child callback
runs (and is *not* recorded).
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Dict, List, Optional

from .contracts import (
    AttributeValue,
    SpanAttributes,
    SpanOptions,
    SpanStatus,
    TelemetryContext,
    TelemetrySpan,
)
from .noop import NOOP_TELEMETRY_CONTEXT


@dataclass(frozen=True)
class RecordedTelemetryEvent:
    name: str
    attributes: "SpanAttributes"


@dataclass(frozen=True)
class RecordedTelemetrySpan:
    id: int
    parent_id: Optional[int]
    name: str
    attributes: "SpanAttributes"
    events: "tuple[RecordedTelemetryEvent, ...]"
    status: SpanStatus
    settled: bool
    end_sequence: Optional[int] = None


@dataclass
class _MutableEvent:
    name: str
    attributes: SpanAttributes


@dataclass
class _MutableSpan:
    id: int
    parent_id: Optional[int]
    name: str
    attributes: SpanAttributes
    events: List[_MutableEvent] = field(default_factory=list)
    status: SpanStatus = field(default_factory=lambda: {"status": "ok"})
    explicit_status: bool = False
    settled: bool = False
    end_sequence: Optional[int] = None


@dataclass
class _State:
    spans: List[_MutableSpan] = field(default_factory=list)
    next_span_id: int = 1
    next_end_sequence: int = 1


def _copy_attribute_value(value: Optional[AttributeValue]) -> Optional[AttributeValue]:
    # Only list/tuple containers are deep-copied (mirrors JS ``Array.isArray``).
    if isinstance(value, (list, tuple)):
        return list(value)
    return value


def _copy_attributes(attributes: Optional[SpanAttributes]) -> SpanAttributes:
    copy: SpanAttributes = {}
    if not attributes:
        return copy
    for name, value in attributes.items():
        if value is not None:
            copy[name] = _copy_attribute_value(value)
    return copy


def _merge_attributes(current: SpanAttributes, attributes: SpanAttributes) -> SpanAttributes:
    merged = _copy_attributes(current)
    # If any value is unreadable, the exception propagates *before* we reassign the
    # stored attributes, so the partially-built merge is discarded (atomic).
    for name, value in attributes.items():
        if value is not None:
            merged[name] = _copy_attribute_value(value)
    return merged


def _copy_status(status: SpanStatus) -> SpanStatus:
    if status.get("status") == "ok":
        return {"status": "ok"}
    error = status.get("error")  # type: ignore[index]
    if error:
        return {"status": "error", "error": {"name": error.get("name"), "message": error.get("message")}}
    return {"status": "error"}


def _automatic_error_status(error: Any) -> SpanStatus:
    try:
        if isinstance(error, BaseException):
            return {
                "status": "error",
                "error": {"name": type(error).__name__, "message": str(error)},
            }
    except Exception:
        # Error inspection is passive; fall through to an error status without details.
        pass
    return {"status": "error"}


def _settle(state: _State, span: _MutableSpan, failed: bool, error: Any = None) -> None:
    if span.settled:
        return
    if failed and not span.explicit_status:
        span.status = _automatic_error_status(error)
    span.settled = True
    span.end_sequence = state.next_end_sequence
    state.next_end_sequence += 1


def _create_span(state: _State, parent: Optional[_MutableSpan], options: SpanOptions) -> _MutableSpan:
    name = options["name"]
    attributes = _copy_attributes(options.get("attributes"))
    return _MutableSpan(
        id=state.next_span_id,
        parent_id=parent.id if parent is not None else None,
        name=name,
        attributes=attributes,
        status={"status": "ok"},
        explicit_status=False,
        settled=False,
    )


async def _await_if_needed(result):
    if inspect.isawaitable(result):
        return await result
    return result


async def _start_in_memory_span(
    state: _State,
    parent: Optional[_MutableSpan],
    options: SpanOptions,
    callback,
) -> Any:
    # A settled parent forwards children to the no-op context: the callback still runs
    # (so results propagate) but nothing is recorded under this context.
    if parent is not None and parent.settled:
        return await NOOP_TELEMETRY_CONTEXT.start_span(options, callback)

    try:
        recorded = _create_span(state, parent, options)
        state.spans.append(recorded)
    except Exception:
        # Reading unreadable options fails passively: run the callback via no-op.
        return await NOOP_TELEMETRY_CONTEXT.start_span(options, callback)

    def make_span() -> TelemetrySpan:
        def start_child(child_options, child_callback):
            return _start_in_memory_span(state, recorded, child_options, child_callback)

        def add_event(name, attributes=None):
            if recorded.settled:
                return
            try:
                recorded.events.append(
                    _MutableEvent(name=name, attributes=_copy_attributes(attributes))
                )
            except Exception:
                # Recording is passive. Ignore malformed or unreadable payloads.
                pass

        def set_attributes(attributes):
            if recorded.settled:
                return
            try:
                recorded.attributes = _merge_attributes(recorded.attributes, attributes)
            except Exception:
                # Recording is passive. Ignore malformed or unreadable payloads.
                pass

        def set_status(status):
            if recorded.settled:
                return
            try:
                recorded.status = _copy_status(status)
                recorded.explicit_status = True
            except Exception:
                # Recording is passive. Ignore malformed or unreadable payloads.
                pass

        class _RecordingSpan:
            def __init__(self, start_child, add_event, set_attributes, set_status):
                self._start_child = start_child
                self._add_event = add_event
                self._set_attributes = set_attributes
                self._set_status = set_status

            def start_span(self, options, callback):
                return self._start_child(options, callback)

            def add_event(self, name, attributes=None):
                return self._add_event(name, attributes)

            def set_attributes(self, attributes):
                return self._set_attributes(attributes)

            def set_status(self, status):
                return self._set_status(status)

        return _RecordingSpan(start_child, add_event, set_attributes, set_status)

    span = make_span()
    try:
        result = callback(span)
    except Exception as error:
        _settle(state, recorded, True, error)
        raise
    try:
        value = await _await_if_needed(result)
    except Exception as error:
        _settle(state, recorded, True, error)
        raise
    _settle(state, recorded, False)
    return value


class InMemoryTelemetryContext:
    """Backend-neutral reference implementation that records spans in process memory.

    Create a fresh instance to isolate tests or independent recording scopes.
    """

    def __init__(self) -> None:
        self._state = _State()

    def start_span(self, options: SpanOptions, callback) -> Awaitable[Any]:
        return _start_in_memory_span(self._state, None, options, callback)

    def get_spans(self) -> List[RecordedTelemetrySpan]:
        """Return detached snapshots in span-start order."""
        snapshots: List[RecordedTelemetrySpan] = []
        for span in self._state.spans:
            events = tuple(
                RecordedTelemetryEvent(name=e.name, attributes=_copy_attributes(e.attributes))
                for e in span.events
            )
            snapshot = RecordedTelemetrySpan(
                id=span.id,
                parent_id=span.parent_id,
                name=span.name,
                attributes=_copy_attributes(span.attributes),
                events=events,
                status=_copy_status(span.status),
                settled=span.settled,
                end_sequence=span.end_sequence,
            )
            snapshots.append(snapshot)
        return snapshots
