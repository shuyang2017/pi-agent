"""RPC-mode test: NDJSON command/event protocol over in-memory streams."""

import asyncio
import io
import json

from pi_tui.rpc_mode import run_rpc
from pi_tui.session_builder import build_session_from_env


def test_rpc_emits_event_lines_for_user_command():
    session = build_session_from_env(mock=True)
    lines = [
        '{"type": "user", "text": "hello"}',
        '{"type": "quit"}',
    ]

    async def reader():
        return lines.pop(0) if lines else None

    out = io.StringIO()
    asyncio.run(run_rpc(session, reader, out))

    emitted = [json.loads(l) for l in out.getvalue().splitlines() if l.strip()]
    types = [e["type"] for e in emitted]
    assert "message_update" in types
    assert any(e.get("delta") == "(mock) Echo: hello" for e in emitted)


def test_rpc_abort_command_does_not_crash():
    session = build_session_from_env(mock=True)
    lines = [
        '{"type": "abort"}',
        '{"type": "quit"}',
    ]

    async def reader():
        return lines.pop(0) if lines else None

    out = io.StringIO()
    asyncio.run(run_rpc(session, reader, out))
    assert out.getvalue().strip() == ""


def test_rpc_invalid_json_reports_error():
    session = build_session_from_env(mock=True)
    lines = ['not-json', '{"type": "quit"}']

    async def reader():
        return lines.pop(0) if lines else None

    out = io.StringIO()
    asyncio.run(run_rpc(session, reader, out))
    emitted = [json.loads(l) for l in out.getvalue().splitlines() if l.strip()]
    assert any(e.get("type") == "error" for e in emitted)
