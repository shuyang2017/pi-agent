import pytest

from pi_telemetry import NOOP_TELEMETRY_CONTEXT


async def test_noop_runs_callback_and_returns_result():
    sentinel = object()
    result = await NOOP_TELEMETRY_CONTEXT.start_span({"name": "x"}, lambda s: sentinel)
    assert result is sentinel


async def test_noop_propagates_synchronous_exception_identity():
    err = ValueError("boom")

    def cb(s):
        raise err

    with pytest.raises(ValueError) as exc:
        await NOOP_TELEMETRY_CONTEXT.start_span({"name": "x"}, cb)
    assert exc.value is err


async def test_noop_propagates_asynchronous_exception_identity():
    err = RuntimeError("async boom")

    async def cb(s):
        raise err

    with pytest.raises(RuntimeError) as exc:
        await NOOP_TELEMETRY_CONTEXT.start_span({"name": "x"}, cb)
    assert exc.value is err


async def test_noop_child_callback_runs_but_records_nothing():
    seen = []

    def parent(s):
        return s.start_span({"name": "c"}, lambda c: seen.append(c) or "ok")

    result = await NOOP_TELEMETRY_CONTEXT.start_span({"name": "p"}, parent)
    assert result == "ok"
    assert len(seen) == 1


async def test_noop_awaitable_callback_result():
    async def cb(s):
        return 5

    assert await NOOP_TELEMETRY_CONTEXT.start_span({"name": "x"}, cb) == 5
