import asyncio

from pi_ai.event_stream import EventStream

from pi_agent_core import AgentEvent


def test_emit_fanout_reaches_all_sinks_without_partitioning():
    """Mirrors the upstream design: a single emit() closure fans out to
    TUI + telemetry + the EventStream. The EventStream remains single-consumer;
    no event is "lost" to one sink. This is the fan-out the user asked about."""

    async def run():
        stream = EventStream[AgentEvent, list](
            is_complete=lambda e: e.type == "agent_end",
            extract_result=lambda e: e.messages if e.messages is not None else [],
        )
        tui_sink: list = []
        telemetry_sink: list = []

        def emit(event: AgentEvent) -> None:
            # Fan-out: each sink gets its own reference; the stream is one consumer.
            tui_sink.append(event)
            telemetry_sink.append(event)
            stream.push(event)

        emit(AgentEvent(type="agent_start"))
        emit(AgentEvent(type="turn_start"))
        emit(AgentEvent(type="tool_execution_start", toolCallId="1", toolName="grep", args={}))
        emit(AgentEvent(type="tool_execution_end", toolCallId="1", toolName="grep", result={}, isError=False))
        emit(AgentEvent(type="agent_end", messages=["m1", "m2"]))
        stream.end(["m1", "m2"])

        consumed = [e async for e in stream]
        result = await stream.result()
        return tui_sink, telemetry_sink, consumed, result

    tui_sink, telemetry_sink, consumed, result = asyncio.run(run())

    expected_types = ["agent_start", "turn_start", "tool_execution_start", "tool_execution_end", "agent_end"]
    assert [e.type for e in tui_sink] == expected_types
    assert [e.type for e in telemetry_sink] == expected_types
    assert [e.type for e in consumed] == expected_types
    # Fan-out did not partition: all three views are identical and complete.
    assert len(tui_sink) == len(telemetry_sink) == len(consumed) == 5
    assert result == ["m1", "m2"]
