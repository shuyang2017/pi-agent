"""AgentSession smoke test: mock LLM issues a tool call, it is executed and
re-injected, then the model finishes with a terminal stop."""

from __future__ import annotations

import asyncio

import pytest

from pi_ai import (
    AssistantMessage,
    AssistantMessageEvent,
    AssistantMessageEventStream,
    Model,
    TextContent,
    ToolCall,
)
from pi_coding_agent.agent_session import create_agent_session
from pi_coding_agent.tools import create_default_tools

pytestmark = pytest.mark.asyncio


def _tool_loop_mock_stream_fn(model, context, options):
    """Turn 1: emit a grep tool call. Turn 2 (after a toolResult is present):
    emit a final text stop."""
    loop = asyncio.get_event_loop()
    stream = AssistantMessageEventStream(loop=loop)
    has_tool_result = any(getattr(m, "role", None) == "toolResult" for m in context.messages)
    out = AssistantMessage(
        role="assistant",
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        stopReason="toolUse" if not has_tool_result else "stop",
    )

    async def _run() -> None:
        stream.push(AssistantMessageEvent(type="start", partial=out))
        if not has_tool_result:
            tc = ToolCall(id="c1", name="grep", arguments={"pattern": "TODO", "path": "."})
            out.content.append(tc)
            stream.push(AssistantMessageEvent(type="toolcall_start", contentIndex=0, partial=out))
            stream.push(AssistantMessageEvent(type="toolcall_end", contentIndex=0, toolCall=tc, partial=out))
            stream.push(AssistantMessageEvent(type="done", reason="toolUse", partial=out))
        else:
            text = "Done searching."
            out.content.append(TextContent(text=text))
            stream.push(AssistantMessageEvent(type="text_start", contentIndex=0, partial=out))
            stream.push(AssistantMessageEvent(type="text_delta", contentIndex=0, delta=text, partial=out))
            stream.push(AssistantMessageEvent(type="text_end", contentIndex=0, content=text, partial=out))
            stream.push(AssistantMessageEvent(type="done", reason="stop", partial=out))
        stream.end()

    loop.create_task(_run())
    return stream


async def test_agent_session_runs_tool_loop(tmp_path):
    (tmp_path / "f.py").write_text("# TODO thing\n")
    model = Model(id="mock", api="mock", provider="mock")
    session = create_agent_session(
        model,
        cwd=str(tmp_path),
        tools=create_default_tools(str(tmp_path)),
        stream_fn=_tool_loop_mock_stream_fn,
    )

    events = await session.send_message("search for TODO")
    # Events should include a tool execution span and a final turn end.
    types = [e.type for e in events]
    assert "tool_execution_start" in types
    assert "tool_execution_end" in types
    assert "agent_end" in types

    # History: user, assistant(toolCall), toolResult(grep), final assistant(text).
    roles = [getattr(m, "role", None) for m in session.messages]
    assert roles == ["user", "assistant", "toolResult", "assistant"]

    tool_result = session.messages[2]
    assert tool_result.toolName == "grep"
    assert "TODO" in tool_result.content[0].text  # grep actually found our file

    final = session.messages[3]
    assert "Done" in final.content[0].text


async def test_agent_session_no_tools_needed(tmp_path):
    """Model can also just answer directly (no tool call)."""
    model = Model(id="mock", api="mock", provider="mock")

    def direct_fn(model, context, options):
        loop = asyncio.get_event_loop()
        stream = AssistantMessageEventStream(loop=loop)
        out = AssistantMessage(role="assistant", content=[], api="mock", provider="mock", model="mock", stopReason="stop")

        async def _run() -> None:
            stream.push(AssistantMessageEvent(type="start", partial=out))
            out.content.append(TextContent(text="hi there"))
            stream.push(AssistantMessageEvent(type="text_start", contentIndex=0, partial=out))
            stream.push(AssistantMessageEvent(type="text_delta", contentIndex=0, delta="hi there", partial=out))
            stream.push(AssistantMessageEvent(type="text_end", contentIndex=0, content="hi there", partial=out))
            stream.push(AssistantMessageEvent(type="done", reason="stop", partial=out))
            stream.end()

        loop.create_task(_run())
        return stream

    session = create_agent_session(model, cwd=str(tmp_path), tools=[], stream_fn=direct_fn)
    await session.send_message("hello")
    assert session.messages[0].role == "user"
    assert session.messages[1].role == "assistant"
    assert "hi there" == session.messages[1].content[0].text
