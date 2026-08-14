"""CLI entrypoint for pi-coding-agent (Python port of coding-agent main).

Reads model configuration from the environment and runs a small async REPL.
When ``PI_MOCK=1`` (or no API key is configured) a canned mock stream function
is used so the agent loop, tool dispatch, and session lifecycle can be
exercised without any network credentials.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any, List, Optional

from pi_ai import (
    AssistantMessage,
    AssistantMessageEvent,
    AssistantMessageEventStream,
    Model,
    TextContent,
)
from pi_ai.event_stream import EventStream

from .agent_session import AgentSession, create_agent_session
from .agent_session_runtime import AgentSessionRuntime

__all__ = ["build_model", "mock_stream_fn", "main"]


def build_model() -> Optional[Model]:
    """Build a :class:`Model` from environment variables.

    Returns ``None`` when mock mode is requested (``PI_MOCK=1``) or when no
    provider/model is configured, signalling the caller to use
    :func:`mock_stream_fn`.
    """
    if os.environ.get("PI_MOCK"):
        return None
    return Model(
        id=os.environ.get("PI_MODEL_ID", "claude-3-5-sonnet"),
        name=os.environ.get("PI_MODEL_NAME", "default"),
        api=os.environ.get("PI_API", "anthropic-messages"),
        provider=os.environ.get("PI_PROVIDER", "anthropic"),
        baseUrl=os.environ.get("PI_BASE_URL", ""),
    )


def _last_user_text(context: Any) -> str:
    for message in reversed(getattr(context, "messages", []) or []):
        if getattr(message, "role", None) == "user":
            content = getattr(message, "content", None)
            if isinstance(content, str):
                return content
            return "".join(
                b.text for b in (content or []) if getattr(b, "text", None)
            )
    return ""


def mock_stream_fn(model: Model, context: Any, options: Any) -> AssistantMessageEventStream:
    """Canned stream function used in mock mode.

    Emits a single assistant text turn ending with ``stop`` — enough to drive
    the agent loop end-to-end without a real LLM.
    """
    loop = asyncio.get_event_loop()
    stream = AssistantMessageEventStream(loop=loop)
    out = AssistantMessage(
        role="assistant",
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        stopReason="stop",
    )

    async def _run() -> None:
        stream.push(AssistantMessageEvent(type="start", partial=out))
        text = f"(mock) Echo: {_last_user_text(context)}"
        out.content.append(TextContent(text=text))
        stream.push(AssistantMessageEvent(type="text_start", contentIndex=0, partial=out))
        stream.push(AssistantMessageEvent(type="text_delta", contentIndex=0, delta=text, partial=out))
        stream.push(AssistantMessageEvent(type="text_end", contentIndex=0, content=text, partial=out))
        stream.push(AssistantMessageEvent(type="done", reason="stop", partial=out))
        stream.end()

    loop.create_task(_run())
    return stream


def _print_agent_events(events: List[Any]) -> None:
    for ev in events:
        t = ev.type
        if t == "message_update" and ev.assistantMessageEvent:
            # Streaming text/thinking deltas ride inside message_update events.
            delta = getattr(ev.assistantMessageEvent, "delta", None)
            if delta:
                sys.stdout.write(delta)
                sys.stdout.flush()
        elif t == "tool_execution_start":
            sys.stdout.write(f"\n[tool:{ev.toolName} {ev.args}]\n")
        elif t == "tool_execution_end" and ev.isError:
            sys.stdout.write("[tool error]\n")
        elif t == "turn_end":
            sys.stdout.write("\n")
    sys.stdout.flush()


async def _repl(session: AgentSession) -> None:
    sys.stdout.write("pi-coding-agent (mock mode). Type /exit to quit.\n")
    while True:
        try:
            line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
        except Exception:
            break
        if line is None:
            break
        text = line.strip()
        if not text:
            continue
        if text in ("/exit", "/quit"):
            break
        events = await session.send_message(text)
        _print_agent_events(events)


def main() -> None:
    cwd = os.getcwd()
    model = build_model()
    stream_fn = None if model is not None else mock_stream_fn
    session = create_agent_session(model or Model(id="mock", api="mock", provider="mock"), cwd=cwd, stream_fn=stream_fn)
    runtime = AgentSessionRuntime(session)
    try:
        asyncio.run(_repl(session))
    finally:
        asyncio.run(runtime.dispose())
