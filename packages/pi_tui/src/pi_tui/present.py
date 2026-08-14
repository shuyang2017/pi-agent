"""Shared rendering helpers for the non-interactive pi_tui front-ends.

These translate :class:`AgentEvent` payloads into plain-text (print mode) and
JSON-line (rpc mode) representations so every front-end renders the *same* event
vocabulary. Interactive (Textual) rendering lives in :mod:`pi_tui.app`.
"""

from __future__ import annotations

import json
import sys
from typing import Any, IO, Optional

from pi_agent_core.types import AgentEvent


def event_to_dict(ev: AgentEvent) -> dict:
    """Serialize an :class:`AgentEvent` to a JSON-friendly dict (rpc protocol)."""
    d: dict = {"type": ev.type}
    if ev.toolName is not None:
        d["toolName"] = ev.toolName
    if ev.toolCallId is not None:
        d["toolCallId"] = ev.toolCallId
    if ev.args is not None:
        d["args"] = ev.args
    if ev.isError:
        d["isError"] = True
    if ev.assistantMessageEvent is not None:
        ame = ev.assistantMessageEvent
        d["eventType"] = ame.type
        delta = getattr(ame, "delta", None)
        if delta:
            d["delta"] = delta
        reason = getattr(ame, "reason", None)
        if reason:
            d["reason"] = reason
    return d


def present_print(ev: AgentEvent, out: IO[str]) -> None:
    """Render one agent event to a text stream (print mode)."""
    t = ev.type
    if t == "message_update" and ev.assistantMessageEvent is not None:
        delta = getattr(ev.assistantMessageEvent, "delta", None)
        if delta:
            out.write(delta)
            out.flush()
    elif t == "tool_execution_start":
        args = ev.args if isinstance(ev.args, (str, int, float, bool)) else json.dumps(ev.args or {})
        out.write(f"\n[tool:{ev.toolName} {args}]\n")
        out.flush()
    elif t == "tool_execution_end":
        if ev.isError:
            out.write("[tool error]\n")
        else:
            out.write("[tool done]\n")
        out.flush()
    elif t == "turn_end":
        out.write("\n")
        out.flush()


def dump_json_line(obj: dict, writer: Any) -> None:
    """Write a single JSON object followed by a newline to an async/textual writer."""
    line = json.dumps(obj, ensure_ascii=False)
    if hasattr(writer, "write") and hasattr(writer, "drain"):
        # Async stream writer (asyncio StreamWriter / in-memory double).
        writer.write((line + "\n").encode())
    else:
        writer.write(line + "\n")
        writer.flush()
