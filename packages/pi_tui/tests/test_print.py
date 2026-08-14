"""Print-mode test: stream a mock turn to an in-memory text sink."""

import asyncio
import io

from pi_tui.print_mode import run_print
from pi_tui.session_builder import build_session_from_env


def test_run_print_renders_mock_echo():
    session = build_session_from_env(mock=True)
    out = io.StringIO()
    asyncio.run(run_print(session, ["hello"], out=out))
    text = out.getvalue()
    assert "[you] hello" in text
    assert "(mock) Echo: hello" in text
