"""pi-tui CLI: choose interactive / print / rpc front-end.

All three modes drive the same :class:`AgentSession` built from the environment
(or forced mock mode), so behavior stays consistent across surfaces.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from .app import CodingAgentApp
from .print_mode import run_print
from .rpc_mode import run_rpc_stdio
from .session_builder import build_session_from_env


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pi-tui", description="pi coding-agent terminal UI")
    p.add_argument(
        "--mode",
        choices=["interactive", "print", "rpc"],
        default="interactive",
        help="front-end: interactive (Textual), print (stdout), rpc (NDJSON)",
    )
    p.add_argument("--cwd", default=".", help="workspace directory for tools")
    p.add_argument(
        "--mock",
        action="store_true",
        help="force mock LLM stream (no network / credentials needed)",
    )
    p.add_argument(
        "--message",
        action="append",
        default=[],
        help="user message(s) for print mode (repeatable); stdin lines if omitted",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    session = build_session_from_env(cwd=args.cwd, mock=args.mock)

    if args.mode == "interactive":
        CodingAgentApp(session).run()
        return 0

    if args.mode == "print":
        messages = list(args.message)
        if not messages:
            messages = [line.rstrip("\n") for line in sys.stdin if line.strip()]
        asyncio.run(run_print(session, messages))
        return 0

    if args.mode == "rpc":
        run_rpc_stdio(session)
        return 0

    return 2  # unreachable


if __name__ == "__main__":
    raise SystemExit(main())
