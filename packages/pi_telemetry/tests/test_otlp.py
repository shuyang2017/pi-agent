import asyncio
import importlib.util

import pytest

from pi_telemetry import create_otlp_telemetry_context

_otel_present = importlib.util.find_spec("opentelemetry") is not None
pytestmark = pytest.mark.skipif(not _otel_present, reason="OpenTelemetry not installed")


def _in_memory_context():
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    ctx = create_otlp_telemetry_context(
        exporter=exporter, span_processor=SimpleSpanProcessor(exporter)
    )
    return ctx


async def test_otlp_requires_configuration():
    with pytest.raises(ValueError):
        create_otlp_telemetry_context()


async def test_otlp_records_spans_with_attributes_and_events():
    ctx = _in_memory_context()

    async def root(span):
        span.set_attributes({"n": 1})
        span.add_event("ev", {"i": 2})
        await span.start_span({"name": "child"}, lambda c: "x")

    await ctx.start_span({"name": "root", "attributes": {"k": "v"}}, root)

    spans = ctx.get_spans()
    assert spans is not None
    by_name = {s["name"]: s for s in spans}
    assert "root" in by_name and "child" in by_name
    assert by_name["root"]["attributes"] == {"k": "v", "n": 1}
    assert by_name["root"]["events"][0]["name"] == "ev"
    assert by_name["child"]["parent_id"] == by_name["root"]["id"]


async def test_otlp_maps_explicit_error_status():
    ctx = _in_memory_context()

    def cb(span):
        span.set_status({"status": "error", "error": {"name": "Boom", "message": "failed"}})
        return "ok"

    result = await ctx.start_span({"name": "err"}, cb)
    assert result == "ok"
    span = [s for s in ctx.get_spans() if s["name"] == "err"][0]
    assert span["status"]["status"] == "error"
    assert span["status"]["error"]["message"] == "failed"


async def test_otlp_automatic_error_status_on_exception():
    ctx = _in_memory_context()

    def cb(span):
        raise ValueError("kaboom")

    with pytest.raises(ValueError):
        await ctx.start_span({"name": "err2"}, cb)
    span = [s for s in ctx.get_spans() if s["name"] == "err2"][0]
    assert span["status"]["status"] == "error"


async def test_otlp_explicit_ok_persists_through_exception():
    ctx = _in_memory_context()
    thrown = RuntimeError("after explicit")

    def cb(span):
        span.set_status({"status": "ok"})
        raise thrown

    with pytest.raises(RuntimeError):
        await ctx.start_span({"name": "ok-exc"}, cb)
    span = [s for s in ctx.get_spans() if s["name"] == "ok-exc"][0]
    assert span["status"]["status"] == "ok"
