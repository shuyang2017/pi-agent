"""Print mode: stream a coding-agent turn to a text sink (stdout by default).

Mirrors the :mod:`pi_tui.present` text vocabulary so the non-interactive and
interactive front-ends stay consistent.
"""

from __future__ import annotations

import sys
from typing import IO, Iterable, List, Optional

from pi_coding_agent.agent_session import AgentSession

from .present import present_print


async def run_print(
    session: AgentSession,
    messages: Iterable[str],
    out: Optional[IO[str]] = None,
) -> None:
    """Run each user message through ``session.stream_turn`` and render live.

    ``messages`` is an iterable of user strings (e.g. argv, or lines from stdin).
    The agent's text deltas, tool calls, and turn boundaries are written to
    ``out`` (defaults to ``sys.stdout``) as they arrive.
    """
    if out is None:
        out = sys.stdout
    for text in messages:
        text = text.strip()
        if not text:
            continue
        out.write(f"\n[you] {text}\n")
        out.flush()
        async for ev in session.stream_turn(text):
            present_print(ev, out)
        out.write("\n")
        out.flush()
