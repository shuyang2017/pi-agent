"""AgentSession + create_agent_session (Python port of coding-agent core).

This is the thin orchestration layer that bridges ``pi_agent_core.agent_loop``
to a conversational session:

* It owns the full message history (``messages``).
* ``send_message`` / ``send_messages`` build an :class:`AgentContext` +
  :class:`AgentLoopConfig` and call :func:`pi_agent_core.agent_loop`, which
  already implements the double-loop, tool dispatch / re-injection, and
  ``length`` self-heal.
* It consumes the resulting ``AgentEvent`` stream and appends the newly
  produced turn to the history.

``convertToLlm`` defaults to identity because our ``AgentMessage`` union *is*
the ``pi_ai`` ``Message`` union (same dataclasses) — no transformation is
needed at the LLM boundary for the common path.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from pi_ai import AssistantMessage, Model, TextContent, UserMessage
from pi_ai.event_stream import EventStream
from pi_agent_core.agent_loop import agent_loop
from pi_agent_core.stream_fn import get_default_stream_fn
from pi_agent_core.types import (
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    AgentTool,
    StreamFn,
)

from .tools import create_default_tools

__all__ = ["AgentSession", "create_agent_session"]


def _identity_convert(messages: List[AgentMessage]) -> Any:
    # AgentMessage is already the pi_ai Message union; no transformation needed.
    return messages


def _content_text(message: Any) -> str:
    """Best-effort extraction of plain text from a message's content."""
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    parts: List[str] = []
    for block in content or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
        else:
            thinking = getattr(block, "thinking", None)
            if isinstance(thinking, str):
                parts.append(thinking)
    return "\n".join(parts)


class AgentSession:
    """Holds conversation state and drives turns through ``agent_loop``."""

    def __init__(
        self,
        model: Model,
        tools: Optional[List[AgentTool]] = None,
        system_prompt: str = "",
        stream_fn: Optional[StreamFn] = None,
        signal: Optional[Any] = None,
        convert_to_llm: Optional[Callable[[List[AgentMessage]], Any]] = None,
    ) -> None:
        self.model = model
        self.tools: List[AgentTool] = tools if tools is not None else []
        self.system_prompt = system_prompt
        self.stream_fn: StreamFn = stream_fn or get_default_stream_fn()
        self.signal = signal
        self.messages: List[AgentMessage] = []
        self._convert = convert_to_llm or _identity_convert

    # -- turn driving -----------------------------------------------------
    async def send_messages(self, prompts: List[AgentMessage]) -> List[AgentEvent]:
        """Run one agent turn for ``prompts``; update history; return events."""
        context = AgentContext(
            systemPrompt=self.system_prompt,
            messages=self.messages,
            tools=self.tools,
        )
        config = AgentLoopConfig(model=self.model, convertToLlm=self._convert)
        stream = agent_loop(prompts, context, config, self.signal, self.stream_fn)
        events: List[AgentEvent] = []
        async for ev in stream:
            events.append(ev)
        new_messages = await stream.result()
        # agent_loop returns only the newly produced turn (prompts + assistant
        # + tool results); extend the full history with it.
        self.messages.extend(new_messages)
        return events

    async def send_message(self, text: str) -> List[AgentEvent]:
        """Convenience wrapper: send a plaintext user message."""
        user = UserMessage(content=[TextContent(text=text)])
        return await self.send_messages([user])

    def send_message_stream(self, text: str) -> EventStream[AgentEvent, List[AgentMessage]]:
        """Lower-level variant returning the raw event stream (no history bookkeeping)."""
        user = UserMessage(content=[TextContent(text=text)])
        context = AgentContext(
            systemPrompt=self.system_prompt,
            messages=self.messages,
            tools=self.tools,
        )
        config = AgentLoopConfig(model=self.model, convertToLlm=self._convert)
        return agent_loop([user], context, config, self.signal, self.stream_fn)

    async def stream_turn(self, text: str):
        """Yield ``AgentEvent``\\ s live for one user turn, then update history.

        Mirrors :meth:`send_message` but exposes the underlying event stream so the
        TUI / print / rpc front-ends can render deltas live. The newly produced
        turn (prompts + assistant + tool results) is appended to history once the
        stream completes, so subsequent turns retain context.
        """
        stream = self.send_message_stream(text)
        async for ev in stream:
            yield ev
        new_messages = await stream.result()
        self.messages.extend(new_messages)

    # -- controls ---------------------------------------------------------
    def abort(self) -> None:
        """Signal an in-flight turn to stop (best-effort)."""
        if self.signal is not None:
            self.signal.set()

    @property
    def last_assistant_text(self) -> str:
        for message in reversed(self.messages):
            if getattr(message, "role", None) == "assistant":
                return _content_text(message)
        return ""


def create_agent_session(
    model: Model,
    cwd: str = ".",
    tools: Optional[List[AgentTool]] = None,
    system_prompt: str = "",
    stream_fn: Optional[StreamFn] = None,
    signal: Optional[Any] = None,
) -> AgentSession:
    """Factory: build a session with default workspace tools when none given."""
    if tools is None:
        tools = create_default_tools(cwd)
    return AgentSession(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        stream_fn=stream_fn,
        signal=signal,
    )
