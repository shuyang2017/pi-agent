import pytest

from pi_telemetry import InMemoryTelemetryContext, RecordedTelemetrySpan


def _ctx():
    return InMemoryTelemetryContext()


async def test_records_single_span_with_ok_status():
    ctx = _ctx()
    result = await ctx.start_span({"name": "a"}, lambda s: 7)
    assert result == 7
    spans = ctx.get_spans()
    assert len(spans) == 1
    assert spans[0].name == "a"
    assert spans[0].status == {"status": "ok"}
    assert spans[0].settled is True
    assert spans[0].parent_id is None


async def test_merges_attributes_and_records_events():
    ctx = _ctx()

    def cb(s):
        s.set_attributes({"count": 1, "overwrite": "middle"})
        s.set_attributes({"count": None, "overwrite": "end"})
        s.add_event("first", {"index": 1, "ignored": None})
        s.add_event("second", {"index": 2})

    await ctx.start_span(
        {"name": "rec", "attributes": {"start": "value", "overwrite": "start", "ignored": None}}, cb
    )
    span = ctx.get_spans()[0]
    assert dict(span.attributes) == {"start": "value", "overwrite": "end", "count": 1}
    assert [(e.name, dict(e.attributes)) for e in span.events] == [
        ("first", {"index": 1}),
        ("second", {"index": 2}),
    ]


async def test_nested_spans_record_parentage_and_end_sequence():
    ctx = _ctx()
    import asyncio

    loop = asyncio.get_event_loop()
    fut = loop.create_future()

    async def parent(s):
        # `create_task` registers the first child (blocked on `fut`) without running
        # it eagerly (unlike JS promise executors); it begins once the loop regains
        # control, i.e. when we `await second` below.
        first = asyncio.create_task(s.start_span({"name": "first-child"}, lambda _s: fut))
        second = s.start_span({"name": "second-child"}, lambda _s: "done")
        assert await second == "done"
        fut.set_result(None)
        await first

    await ctx.start_span({"name": "parent"}, parent)
    spans = ctx.get_spans()
    by_name = {s.name: s for s in spans}
    assert by_name["parent"].parent_id is None
    assert by_name["first-child"].parent_id == by_name["parent"].id
    assert by_name["second-child"].parent_id == by_name["parent"].id
    assert by_name["second-child"].end_sequence < by_name["first-child"].end_sequence
    assert by_name["first-child"].end_sequence < by_name["parent"].end_sequence


async def test_settled_span_is_inert_but_child_callback_still_runs():
    ctx = _ctx()
    captured = {}

    def capture(s):
        captured["s"] = s

    await ctx.start_span({"name": "settled", "attributes": {"value": "initial"}}, capture)
    s = captured["s"]
    s.set_attributes({"value": "late"})
    s.add_event("late", {"value": True})
    s.set_status({"status": "error"})

    child_result = await s.start_span({"name": "late-child"}, lambda c: 7)
    assert child_result == 7

    spans = ctx.get_spans()
    assert len(spans) == 1
    assert dict(spans[0].attributes) == {"value": "initial"}
    assert list(spans[0].events) == []
    assert spans[0].status == {"status": "ok"}


async def test_explicit_status_takes_precedence_over_automatic_error():
    ctx = _ctx()
    thrown = RuntimeError("after explicit")

    def cb(s):
        s.set_status({"status": "ok"})
        raise thrown

    with pytest.raises(RuntimeError) as exc:
        await ctx.start_span({"name": "x"}, cb)
    assert exc.value is thrown
    assert ctx.get_spans()[0].status == {"status": "ok"}


async def test_automatic_error_status_from_exception():
    ctx = _ctx()

    def cb(s):
        raise ValueError("kaboom")

    with pytest.raises(ValueError):
        await ctx.start_span({"name": "x"}, cb)
    assert ctx.get_spans()[0].status == {
        "status": "error",
        "error": {"name": "ValueError", "message": "kaboom"},
    }


async def test_get_spans_returns_detached_snapshots():
    ctx = _ctx()
    await ctx.start_span({"name": "a", "attributes": {"k": "v"}}, lambda s: None)
    spans = ctx.get_spans()
    spans[0].attributes["k"] = "mutated"
    assert ctx.get_spans()[0].attributes["k"] == "v"
