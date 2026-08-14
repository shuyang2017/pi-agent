"""SSE line decoder (Python port of the SSE plumbing in packages/ai/src/api/anthropic-messages.ts).

Upstream reads raw bytes, decodes incrementally with a TextDecoder, and splits
on line breaks (``\\r``, ``\\n``, ``\\r\\n``). httpx already exposes decoded
lines via ``response.aiter_lines()``, so this port consumes an async iterator of
*lines* and reconstructs Server-Sent-Events from the ``event:`` / ``data:``
fields — behaviour identical to the upstream ``decodeSseLine`` + ``flushSseEvent``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import AsyncIterator, List, Optional


@dataclass
class ServerSentEvent:
    event: Optional[str]
    data: str
    raw: List[str] = field(default_factory=list)


def _flush(state: dict) -> Optional[ServerSentEvent]:
    if not state["event"] and not state["data"]:
        return None
    event = ServerSentEvent(
        event=state["event"],
        data="\n".join(state["data"]),
        raw=list(state["raw"]),
    )
    state["event"] = None
    state["data"] = []
    state["raw"] = []
    return event


async def iterate_sse_messages(
    lines: AsyncIterator[str],
    abort_event: Optional[asyncio.Event] = None,
) -> AsyncIterator[ServerSentEvent]:
    state = {"event": None, "data": [], "raw": []}

    async for line in lines:
        if abort_event is not None and abort_event.is_set():
            raise asyncio.CancelledError("Request was aborted")

        if line == "":
            ev = _flush(state)
            if ev is not None:
                yield ev
            continue

        state["raw"].append(line)
        if line.startswith(":"):
            continue

        if ":" in line:
            idx = line.index(":")
            field = line[:idx]
            value = line[idx + 1 :]
            if value.startswith(" "):
                value = value[1:]
        else:
            field = line
            value = ""

        if field == "event":
            state["event"] = value
        elif field == "data":
            state["data"].append(value)

    trailing = _flush(state)
    if trailing is not None:
        yield trailing


__all__ = ["ServerSentEvent", "iterate_sse_messages"]
