"""Unit tests for pi_tui.present (shared event→text / event→dict helpers)."""

import io
import json

from pi_ai import AssistantMessage
from pi_ai.types import AssistantMessageEvent
from pi_agent_core.types import AgentEvent

from pi_tui.present import dump_json_line, event_to_dict, present_print


def _ame(type: str, **kw) -> AssistantMessageEvent:
    partial = kw.pop(
        "partial",
        AssistantMessage(role="assistant", content=[], api="a", provider="b", model="c"),
    )
    return AssistantMessageEvent(type=type, partial=partial, **kw)


def test_event_to_dict_text_delta():
    ev = AgentEvent(
        type="message_update",
        assistantMessageEvent=_ame("text_delta", delta="hi"),
    )
    assert event_to_dict(ev) == {
        "type": "message_update",
        "eventType": "text_delta",
        "delta": "hi",
    }


def test_event_to_dict_tool_start():
    ev = AgentEvent(
        type="tool_execution_start",
        toolName="grep",
        toolCallId="t1",
        args={"pattern": "x"},
    )
    d = event_to_dict(ev)
    assert d["type"] == "tool_execution_start"
    assert d["toolName"] == "grep"
    assert d["toolCallId"] == "t1"
    assert d["args"] == {"pattern": "x"}
    assert "delta" not in d


def test_present_print_writes_delta():
    out = io.StringIO()
    present_print(
        AgentEvent(
            type="message_update",
            assistantMessageEvent=_ame("text_delta", delta="Hi"),
        ),
        out,
    )
    assert out.getvalue() == "Hi"


def test_present_print_tool_lines():
    out = io.StringIO()
    present_print(
        AgentEvent(type="tool_execution_start", toolName="grep", args="x"),
        out,
    )
    present_print(AgentEvent(type="tool_execution_end", isError=False), out)
    v = out.getvalue()
    assert "[tool:grep x]" in v
    assert "[tool done]" in v


def test_dump_json_line_text_writer():
    out = io.StringIO()
    dump_json_line({"type": "x", "delta": "ab"}, out)
    assert json.loads(out.getvalue()) == {"type": "x", "delta": "ab"}
