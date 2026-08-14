"""RPC mode: newline-delimited JSON command/event protocol.

The host process writes command lines to stdin and reads agent event lines from
stdout. Every event is the same :func:`pi_tui.present.event_to_dict` shape used
by the other front-ends, so a remote driver sees an identical vocabulary.

Command lines (one JSON object per line)::

    {"type": "user",   "text": "list the files"}
    {"type": "abort"}
    {"type": "quit"}

Event lines mirror :class:`AgentEvent` (``type`` plus optional ``toolName``,
``toolCallId``, ``args``, ``isError``, ``eventType``, ``delta``, ``reason``).
"""

from __future__ import annotations

import json
import sys
from typing import Any, Awaitable, Callable, Optional

from pi_coding_agent.agent_session import AgentSession

from .present import dump_json_line, event_to_dict

# A reader yields the next raw line (str) or ``None`` on EOF.
Reader = Callable[[], Awaitable[Optional[str]]]
# A writer accepts a JSON-ready dict and emits one line.
Writer = Any


async def run_rpc(
    session: AgentSession,
    reader: Reader,
    writer: Writer,
    emit: bool = True,
) -> None:
    """Drive ``session`` from an NDJSON command stream until ``quit``/EOF.

    ``reader`` returns the next command line or ``None`` at EOF; ``writer`` is
    passed each event dict via :func:`pi_tui.present.dump_json_line`. When
    ``emit`` is ``False`` no event lines are written (useful for tests that only
    need the side effects on ``session``).
    """
    while True:
        line = await reader()
        if line is None:
            return
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except json.JSONDecodeError:
            if emit:
                dump_json_line({"type": "error", "message": "invalid JSON command"}, writer)
            continue

        ctype = cmd.get("type")
        if ctype == "quit":
            return
        if ctype == "abort":
            session.abort()
            continue
        if ctype != "user":
            if emit:
                dump_json_line(
                    {"type": "error", "message": f"unknown command: {ctype!r}"}, writer
                )
            continue

        text = (cmd.get("text") or "").strip()
        if not text:
            continue
        async for ev in session.stream_turn(text):
            if emit:
                dump_json_line(event_to_dict(ev), writer)


async def _stdin_reader() -> Optional[str]:
    """Async reader yielding lines from stdin (EOF -> None)."""
    import asyncio

    loop = asyncio.get_event_loop()
    line = await loop.run_in_executor(None, sys.stdin.readline)
    if line == "":
        return None
    return line


def run_rpc_stdio(session: AgentSession) -> None:
    """Blocking entry point: read commands from stdin, write events to stdout."""
    import asyncio

    asyncio.run(run_rpc(session, _stdin_reader, sys.stdout))
