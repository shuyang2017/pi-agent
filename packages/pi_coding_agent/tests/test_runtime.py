"""Tests for AgentSessionRuntime lifecycle + JSONL session store."""

from __future__ import annotations

import asyncio

import pytest

from pi_ai import AssistantMessage, Model, TextContent, ToolResultMessage, UserMessage
from pi_coding_agent.agent_session import create_agent_session
from pi_coding_agent.agent_session_runtime import (
    AgentSessionRuntime,
    load_messages,
    save_messages,
)

pytestmark = pytest.mark.asyncio


def _seed_messages():
    return [
        UserMessage(content=[TextContent(text="a")]),
        AssistantMessage(role="assistant", content=[TextContent(text="1")], api="mock", provider="mock", model="m", stopReason="stop"),
        UserMessage(content=[TextContent(text="b")]),
        AssistantMessage(role="assistant", content=[TextContent(text="2")], api="mock", provider="mock", model="m", stopReason="stop"),
    ]


async def test_new_session_clears(tmp_path):
    session = create_agent_session(Model(id="m", api="mock", provider="mock"), cwd=str(tmp_path))
    runtime = AgentSessionRuntime(session)
    session.messages = _seed_messages()
    await runtime.new_session()
    assert session.messages == []


async def test_fork_truncates(tmp_path):
    session = create_agent_session(Model(id="m", api="mock", provider="mock"), cwd=str(tmp_path))
    runtime = AgentSessionRuntime(session)
    session.messages = _seed_messages()
    await runtime.fork("2")
    assert len(session.messages) == 2
    assert session.messages[0].content[0].text == "a"


async def test_save_load_roundtrip(tmp_path):
    path = str(tmp_path / "sess.jsonl")
    msgs = _seed_messages()
    save_messages(msgs, path)
    loaded = load_messages(path)
    assert len(loaded) == 4
    assert loaded[0].content[0].text == "a"
    assert loaded[1].content[0].text == "1"
    assert loaded[3].role == "assistant"

    # A tool result message also survives the round-trip.
    with_tool = msgs + [
        ToolResultMessage(role="toolResult", toolCallId="c1", toolName="grep", content=[TextContent(text="x")], isError=False)
    ]
    p2 = str(tmp_path / "sess2.jsonl")
    save_messages(with_tool, p2)
    loaded2 = load_messages(p2)
    assert loaded2[-1].toolName == "grep"
    assert loaded2[-1].content[0].text == "x"


async def test_import_from_jsonl(tmp_path):
    session = create_agent_session(Model(id="m", api="mock", provider="mock"), cwd=str(tmp_path))
    runtime = AgentSessionRuntime(session)
    session.messages = _seed_messages()
    path = str(tmp_path / "export.jsonl")
    save_messages(session.messages, path)
    session.messages = []
    await runtime.import_from_jsonl(path)
    assert len(session.messages) == 4


async def test_dispose_is_safe(tmp_path):
    session = create_agent_session(Model(id="m", api="mock", provider="mock"), cwd=str(tmp_path))
    runtime = AgentSessionRuntime(session)
    await runtime.dispose()  # must not raise
    assert session.messages == []
