"""Session construction shared by the pi_tui front-ends.

Reuses the Phase 3 ``pi_coding_agent`` session factory and mock stream function so
the TUI, print, and rpc modes all drive the exact same :class:`AgentSession`.
"""

from __future__ import annotations

import os
from typing import Optional

from pi_ai import Model
from pi_coding_agent.agent_session import AgentSession, create_agent_session
from pi_coding_agent.cli import build_model, mock_stream_fn


def build_session_from_env(cwd: str = ".", mock: bool = False) -> AgentSession:
    """Build an :class:`AgentSession` from the environment (or forced mock mode).

    Honors the same variables as ``pi-coding-agent`` (``PI_MODEL_ID``, ``PI_API``,
    ``PI_PROVIDER``, ``PI_BASE_URL``). With ``mock=True`` (or no credentials) a
    canned stream function is used so the whole stack runs without network access.
    """
    if mock:
        os.environ["PI_MOCK"] = "1"
    model = build_model()
    stream_fn = None if model is not None else mock_stream_fn
    # AgentSession requires a concrete Model object; in mock mode the placeholder
    # is only used for metadata (api/provider/model id) — mock_stream_fn ignores it.
    effective_model = model or Model(id="mock", api="mock", provider="mock")
    return create_agent_session(
        effective_model, cwd=cwd, stream_fn=stream_fn
    )
