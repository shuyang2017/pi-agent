import asyncio

import pytest

from pi_ai import AssistantMessage, AssistantMessageEvent, AssistantMessageEventStream
from pi_ai.event_stream import EventStream


async def _collect(stream):
    return [e async for e in stream]


def test_event_stream_basic_order_and_result():
    async def run():
        stream = EventStream[str, str](
            is_complete=lambda e: e == "end",
            extract_result=lambda e: "FINAL",
        )
        stream.push("a")
        stream.push("b")
        stream.push("end")
        stream.end()
        events = await _collect(stream)
        result = await stream.result()
        return events, result

    events, result = asyncio.run(run())
    assert events == ["a", "b", "end"]
    assert result == "FINAL"


def test_event_stream_waits_for_slow_producer():
    async def run():
        stream = EventStream[int, int](
            is_complete=lambda e: e == -1,
            extract_result=lambda e: 99,
        )

        async def producer():
            await asyncio.sleep(0.01)
            stream.push(1)
            stream.push(2)
            stream.push(-1)
            stream.end()

        task = asyncio.create_task(producer())
        events = await _collect(stream)
        await task
        return events

    assert asyncio.run(run()) == [1, 2, -1]


def test_assistant_stream_drops_events_after_complete():
    stream = AssistantMessageEventStream()
    stream.push(AssistantMessageEvent(type="start", partial=AssistantMessage()))
    stream.push(AssistantMessageEvent(type="done", reason="stop", partial=AssistantMessage(stopReason="stop")))
    # pushes after completion are ignored
    stream.push(AssistantMessageEvent(type="text_delta", contentIndex=0, delta="x", partial=AssistantMessage()))
    assert stream._done is True


def test_assistant_stream_result_is_final_message():
    async def run():
        stream = AssistantMessageEventStream()
        final = AssistantMessage(stopReason="stop", content=[])
        stream.push(AssistantMessageEvent(type="done", reason="stop", partial=final))
        stream.end()
        return await stream.result()

    assert asyncio.run(run()).stopReason == "stop"
