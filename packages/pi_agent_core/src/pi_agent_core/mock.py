"""Reusable mock streaming LLM (used by tests and as a porting reference).

Builds an ``AssistantMessageEventStream`` from a scripted event sequence, exactly
like the upstream adapters do, but without any network call. Also provides a
stateful ``grep_demo_stream_fn`` that drives the acceptance scenario:

    user prompt -> assistant calls grep -> tool result re-injected -> final answer
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, List, Optional

from pi_ai import (
    AssistantMessage,
    AssistantMessageEvent,
    AssistantMessageEventStream,
    Context,
    Model,
    TextContent,
    ToolCall,
    Usage,
)
from pi_ai.event_stream import AssistantMessageEventStream as _S


def make_assistant_stream(event_seq: List[AssistantMessageEvent], loop: Optional[asyncio.AbstractEventLoop] = None):
    """Produce a stream that replays ``event_seq`` (terminal done/error included)."""
    loop = loop or asyncio.get_event_loop()
    stream = _S(loop=loop)

    async def _produce() -> None:
        for ev in event_seq:
            stream.push(ev)
        stream.end()

    loop.create_task(_produce())
    return stream


def _usage() -> Usage:
    return Usage(input=0, output=0, cacheRead=0, cacheWrite=0, cacheWrite1h=0, reasoning=0, totalTokens=0)


def grep_tool_call_assistant() -> AssistantMessage:
    out = AssistantMessage(
        role="assistant",
        content=[ToolCall(id="call_1", name="grep", arguments={"pattern": "TODO", "path": "."})],
        api="mock",
        provider="mock",
        model="mock-model",
        usage=_usage(),
        stopReason="toolUse",
        timestamp=0,
    )
    return out


def final_text_assistant(text: str) -> AssistantMessage:
    out = AssistantMessage(
        role="assistant",
        content=[TextContent(text=text)],
        api="mock",
        provider="mock",
        model="mock-model",
        usage=_usage(),
        stopReason="stop",
        timestamp=0,
    )
    return out


def grep_demo_stream_fn(model: Model, context: Context, options: Any = None) -> AssistantMessageEventStream:
    """Stateful mock: first turn calls grep, subsequent turns (after a tool result) answer."""
    loop = asyncio.get_event_loop()
    stream = _S(loop=loop)

    async def _produce() -> None:
        last = context.messages[-1] if context.messages else None
        if last is not None and getattr(last, "role", None) == "toolResult":
            out = final_text_assistant("grep 完成：在 3 个文件中发现 TODO 标记。")
            stream.push(AssistantMessageEvent(type="start", partial=out))
            stream.push(AssistantMessageEvent(type="text_start", contentIndex=0, partial=out))
            stream.push(
                AssistantMessageEvent(type="text_delta", contentIndex=0, delta="grep 完成：在 3 个文件中发现 TODO 标记。", partial=out)
            )
            stream.push(
                AssistantMessageEvent(type="text_end", contentIndex=0, content="grep 完成：在 3 个文件中发现 TODO 标记。", partial=out)
            )
            stream.push(AssistantMessageEvent(type="done", reason="stop", partial=out))
        else:
            out = grep_tool_call_assistant()
            stream.push(AssistantMessageEvent(type="start", partial=out))
            stream.push(AssistantMessageEvent(type="toolcall_start", contentIndex=0, partial=out))
            stream.push(AssistantMessageEvent(type="toolcall_end", contentIndex=0, toolCall=out.content[0], partial=out))
            stream.push(AssistantMessageEvent(type="done", reason="toolUse", partial=out))
        stream.end()

    loop.create_task(_produce())
    return stream


def error_stream_fn(model: Model, context: Context, options: Any = None) -> AssistantMessageEventStream:
    loop = asyncio.get_event_loop()
    stream = _S(loop=loop)

    async def _produce() -> None:
        out = final_text_assistant("")
        out.stopReason = "error"
        out.errorMessage = "mock failure"
        stream.push(AssistantMessageEvent(type="start", partial=out))
        stream.push(AssistantMessageEvent(type="error", reason="error", partial=out, error=out))
        stream.end()

    loop.create_task(_produce())
    return stream


def length_truncated_stream_fn(model: Model, context: Context, options: Any = None) -> AssistantMessageEventStream:
    """First turn emits a tool call that stops with reason "length" (truncated
    arguments). The agent loop self-heals by failing the truncated tool call and
    re-injecting an error tool result; on the next turn (after that tool result)
    we emit a final ``stop`` answer so the loop terminates."""
    loop = asyncio.get_event_loop()
    stream = _S(loop=loop)

    async def _produce() -> None:
        last = context.messages[-1] if context.messages else None
        if last is not None and getattr(last, "role", None) == "toolResult":
            out = final_text_assistant(
                "The previous tool call was truncated; I re-issue it with complete arguments."
            )
            stream.push(AssistantMessageEvent(type="start", partial=out))
            stream.push(AssistantMessageEvent(type="text_start", contentIndex=0, partial=out))
            stream.push(
                AssistantMessageEvent(
                    type="text_delta",
                    contentIndex=0,
                    delta="The previous tool call was truncated; I re-issue it with complete arguments.",
                    partial=out,
                )
            )
            stream.push(
                AssistantMessageEvent(
                    type="text_end",
                    contentIndex=0,
                    content="The previous tool call was truncated; I re-issue it with complete arguments.",
                    partial=out,
                )
            )
            stream.push(AssistantMessageEvent(type="done", reason="stop", partial=out))
        else:
            out = AssistantMessage(
                role="assistant",
                content=[ToolCall(id="call_x", name="grep", arguments={})],
                api="mock",
                provider="mock",
                model="mock-model",
                usage=_usage(),
                stopReason="length",
                timestamp=0,
            )
            stream.push(AssistantMessageEvent(type="start", partial=out))
            stream.push(AssistantMessageEvent(type="toolcall_start", contentIndex=0, partial=out))
            stream.push(AssistantMessageEvent(type="toolcall_end", contentIndex=0, toolCall=out.content[0], partial=out))
            stream.push(AssistantMessageEvent(type="done", reason="length", partial=out))
        stream.end()

    loop.create_task(_produce())
    return stream


__all__ = [
    "make_assistant_stream",
    "grep_demo_stream_fn",
    "error_stream_fn",
    "length_truncated_stream_fn",
    "grep_tool_call_assistant",
    "final_text_assistant",
]
