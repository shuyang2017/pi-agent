"""Conformance harness for telemetry adapter implementations.

Mirrors upstream ``testing/conformance.ts``: a runner-independent set of cases that any
:class:`TelemetryContext` recording adapter must satisfy. Each case builds a *fresh*
fixture via the supplied factory and exercises one invariant (callback lifecycle,
rejection identity, explicit-status precedence, attribute merge, atomic failure,
settled-inert, parentage, and passive handling of unreadable payloads).

Only adapters that can produce a normalized span snapshot via ``get_spans`` are
conformance targets (the in-memory reference adapter, and the OTLP adapter when backed
by an in-memory exporter). The no-op context is exercised separately.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List, Optional

from ..contracts import SpanAttributes, SpanOptions, TelemetryContext


@dataclass
class TelemetryAdapterFixture:
    context: TelemetryContext
    get_spans: Callable[[], Any]


TelemetryAdapterFixtureFactory = Callable[[], Awaitable[TelemetryAdapterFixture]]


@dataclass
class TelemetryAdapterConformanceCase:
    group: str
    name: str
    run: Callable[[], Awaitable[None]]


class _PoisonDict(dict):
    """A mapping that raises on every access (mirrors an unreadable JS proxy)."""

    def __getitem__(self, key):  # noqa: D401
        raise RuntimeError("read")

    def get(self, key, default=None):
        raise RuntimeError("read")

    def items(self):
        raise RuntimeError("read")

    def keys(self):
        raise RuntimeError("read")

    def __iter__(self):
        raise RuntimeError("read")


class _PoisonList(list):
    """A list whose iteration raises (mirrors an unreadable array proxy)."""

    def __iter__(self):
        raise RuntimeError("read")


def _await(value):
    if inspect.isawaitable(value):
        return value
    return _ready(value)


async def _ready(value):
    return value


async def _get_spans(fixture: TelemetryAdapterFixture) -> List[Any]:
    return await _await(fixture.get_spans())


def _find_span(spans: List[Any], name: str) -> Any:
    for span in spans:
        if getattr(span, "name", None) == name:
            return span
    raise AssertionError(f"Expected recorded span {name!r}")


def create_telemetry_adapter_conformance(
    factory: TelemetryAdapterFixtureFactory,
) -> List[TelemetryAdapterConformanceCase]:
    def make(group: str, name: str, test):
        async def run() -> None:
            fixture = await factory()
            await test(fixture)

        return TelemetryAdapterConformanceCase(group=group, name=name, run=run)

    cases: List[TelemetryAdapterConformanceCase] = []

    cases.append(
        make(
            "callback lifecycle",
            "admits once synchronously and preserves the result",
            _case_callback_lifecycle,
        )
    )
    cases.append(
        make(
            "callback lifecycle",
            "preserves synchronous and asynchronous rejection values",
            _case_rejection_identity,
        )
    )
    cases.append(
        make("status", "uses last explicit status without automatic overwrite", _case_status_precedence)
    )
    cases.append(
        make("recording", "merges attributes and records ordered events", _case_merge_attributes)
    )
    cases.append(
        make("recording", "ignores failed attribute calls atomically", _case_atomic_attributes)
    )
    cases.append(
        make("recording", "makes calls after settlement inert", _case_settled_inert)
    )
    cases.append(
        make("parentage", "records nested and concurrent child relationships", _case_parentage)
    )
    cases.append(
        make("passivity", "suppresses unreadable telemetry payload failures", _case_unreadable_payloads)
    )
    cases.append(
        make("passivity", "ignores failed status calls atomically", _case_atomic_status)
    )
    return cases


# ---------------------------------------------------------------------------
# Individual cases
# ---------------------------------------------------------------------------


async def _case_callback_lifecycle(fixture: TelemetryAdapterFixture) -> None:
    admitted = False
    calls = 0
    expected = {"value": 42}

    def callback(span):
        nonlocal admitted, calls
        admitted = True
        calls += 1
        return expected

    result = await fixture.context.start_span({"name": "success"}, callback)
    assert admitted is True
    assert calls == 1
    assert result is expected

    spans = await _get_spans(fixture)
    span = _find_span(spans, "success")
    assert dict(span.status) == {"status": "ok"}
    assert span.settled is True


async def _case_rejection_identity(fixture: TelemetryAdapterFixture) -> None:
    sync_error = ValueError("sync")

    def sync_callback(span):
        raise sync_error

    try:
        await fixture.context.start_span({"name": "sync-error"}, sync_callback)
        raise AssertionError("expected sync-error to reject")
    except Exception as error:  # noqa: BLE001
        assert error is sync_error

    async def async_callback(span):
        raise RuntimeError("async")

    try:
        await fixture.context.start_span({"name": "async-error"}, async_callback)
        raise AssertionError("expected async-error to reject")
    except RuntimeError as error:
        assert str(error) == "async"

    # Poison error whose inspection raises -- exercising passive automatic status.
    class PoisonError(Exception):
        def __str__(self):
            raise RuntimeError("inspect")

    poison = PoisonError("poison")

    def poison_callback(span):
        raise poison

    try:
        await fixture.context.start_span({"name": "poison-error"}, poison_callback)
        raise AssertionError("expected poison-error to reject")
    except Exception as error:  # noqa: BLE001
        assert error is poison

    spans = await _get_spans(fixture)
    for name in ("sync-error", "async-error", "poison-error"):
        assert _find_span(spans, name).status["status"] == "error"


async def _case_status_precedence(fixture: TelemetryAdapterFixture) -> None:
    await fixture.context.start_span({"name": "last-status"}, lambda span: (
        span.set_status({"status": "error", "error": {"name": "Expected", "message": "first"}}),
        span.set_status({"status": "ok"}),
    ))

    thrown = RuntimeError("after explicit status")

    def explicit_before_throw(span):
        span.set_status({"status": "ok"})
        raise thrown

    try:
        await fixture.context.start_span({"name": "explicit-before-throw"}, explicit_before_throw)
        raise AssertionError("expected explicit-before-throw to reject")
    except RuntimeError as error:
        assert error is thrown

    rejected = RuntimeError("after async explicit status")

    def explicit_before_rejection(span):
        span.set_status({"status": "error", "error": {"name": "Expected", "message": "async failure"}})
        return _reject(rejected)

    try:
        await fixture.context.start_span({"name": "explicit-before-rejection"}, explicit_before_rejection)
        raise AssertionError("expected explicit-before-rejection to reject")
    except RuntimeError as error:
        assert error is rejected

    await fixture.context.start_span(
        {"name": "expected-failure"},
        lambda span: (
            span.set_status({"status": "error", "error": {"name": "Expected", "message": "returned failure"}}),
            {"ok": False},
        ),
    )

    spans = await _get_spans(fixture)
    assert dict(_find_span(spans, "last-status").status) == {"status": "ok"}
    assert dict(_find_span(spans, "explicit-before-throw").status) == {"status": "ok"}
    assert dict(_find_span(spans, "explicit-before-rejection").status) == {
        "status": "error",
        "error": {"name": "Expected", "message": "async failure"},
    }
    assert dict(_find_span(spans, "expected-failure").status) == {
        "status": "error",
        "error": {"name": "Expected", "message": "returned failure"},
    }


async def _reject(error: Exception):
    raise error


async def _case_merge_attributes(fixture: TelemetryAdapterFixture) -> None:
    def callback(span):
        span.set_attributes({"count": 1, "overwrite": "middle"})
        span.set_attributes({"count": None, "overwrite": "end"})
        span.add_event("first", {"index": 1, "ignored": None})
        span.add_event("second", {"index": 2})

    await fixture.context.start_span(
        {"name": "recording", "attributes": {"start": "value", "overwrite": "start", "ignored": None}},
        callback,
    )

    span = _find_span(await _get_spans(fixture), "recording")
    assert dict(span.attributes) == {"start": "value", "overwrite": "end", "count": 1}
    assert [(e.name, dict(e.attributes)) for e in span.events] == [
        ("first", {"index": 1}),
        ("second", {"index": 2}),
    ]


async def _case_atomic_attributes(fixture: TelemetryAdapterFixture) -> None:
    poison = _PoisonList(["value"])

    def callback(span):
        attributes: SpanAttributes = {"partial": "must not survive", "unreadable": poison}
        span.set_attributes(attributes)

    await fixture.context.start_span(
        {"name": "atomic-attributes", "attributes": {"retained": "value"}}, callback
    )

    span = _find_span(await _get_spans(fixture), "atomic-attributes")
    assert dict(span.attributes) == {"retained": "value"}


async def _case_settled_inert(fixture: TelemetryAdapterFixture) -> None:
    captured = {}

    def capture(span):
        captured["span"] = span

    await fixture.context.start_span(
        {"name": "settled", "attributes": {"value": "initial"}}, capture
    )
    span = captured["span"]

    # All mutations after settlement are inert on the parent.
    span.set_attributes({"value": "late"})
    span.add_event("late", {"value": True})
    span.set_status({"status": "error"})

    child_admitted = False

    def child(span):
        nonlocal child_admitted
        child_admitted = True
        return 7

    child_result = await span.start_span({"name": "late-child"}, child)
    assert child_admitted is True
    assert child_result == 7

    spans = await _get_spans(fixture)
    assert len(spans) == 1
    assert dict(spans[0].attributes) == {"value": "initial"}
    assert list(spans[0].events) == []
    assert dict(spans[0].status) == {"status": "ok"}


async def _case_parentage(fixture: TelemetryAdapterFixture) -> None:
    import asyncio

    loop = asyncio.get_event_loop()
    fut = loop.create_future()

    async def parent(span):
        # `create_task` registers (but does not eagerly run) the first child, which
        # blocks on `fut`; it begins once the loop regains control at `await second`.
        first_task = asyncio.create_task(
            span.start_span({"name": "first-child"}, lambda _s: fut)
        )
        second = span.start_span({"name": "second-child"}, lambda _s: "done")
        assert await second == "done"
        fut.set_result(None)
        await first_task

    await fixture.context.start_span({"name": "parent"}, parent)

    spans = await _get_spans(fixture)
    parent = _find_span(spans, "parent")
    first = _find_span(spans, "first-child")
    second = _find_span(spans, "second-child")
    assert parent.parent_id is None
    assert first.parent_id == parent.id
    assert second.parent_id == parent.id
    assert second.end_sequence is not None
    assert first.end_sequence is not None
    assert parent.end_sequence is not None
    assert second.end_sequence < first.end_sequence < parent.end_sequence


async def _await_gate(gate: dict):
    import asyncio

    fut = asyncio.get_event_loop().create_future()
    gate["future"] = fut
    gate["release"] = lambda: fut.set_result(None)
    await fut


async def _case_unreadable_payloads(fixture: TelemetryAdapterFixture) -> None:
    calls = 0
    options = _PoisonDict({"name": "unreadable-options", "attributes": {"secret": "value"}})

    def callback(span):
        nonlocal calls
        calls += 1
        return 9

    result = await fixture.context.start_span(options, callback)  # type: ignore[arg-type]
    assert calls == 1
    assert result == 9
    assert await _get_spans(fixture) == []

    def recording(span):
        attributes = _PoisonDict({"secret": "value"})
        status = _PoisonDict({"status": "ok"})
        span.set_attributes(attributes)
        span.add_event("unreadable-event", attributes)
        span.set_status(status)

    await fixture.context.start_span({"name": "unreadable-recording"}, recording)

    recorded = await _get_spans(fixture)
    assert len(recorded) == 1
    assert dict(recorded[0].attributes) == {}
    assert list(recorded[0].events) == []
    assert dict(recorded[0].status) == {"status": "ok"}


async def _case_atomic_status(fixture: TelemetryAdapterFixture) -> None:
    rejection = RuntimeError("rejected after unreadable status")

    def callback(span):
        status = _PoisonDict({"status": "ok"})
        span.set_status(status)
        return _reject(rejection)

    try:
        await fixture.context.start_span({"name": "unreadable-status"}, callback)
        raise AssertionError("expected unreadable-status to reject")
    except RuntimeError as error:
        assert error is rejection

    assert _find_span(await _get_spans(fixture), "unreadable-status").status["status"] == "error"
