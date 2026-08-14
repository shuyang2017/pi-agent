import asyncio

from pi_ai import AssistantMessage, AssistantMessageEvent, Model, TextContent, ToolCall, ToolResultMessage, Usage, UserMessage
from pi_ai.event_stream import AssistantMessageEventStream

from pi_agent_core import (
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentTool,
    AgentToolResult,
    agent_loop,
)
from pi_agent_core.mock import (
    error_stream_fn,
    grep_demo_stream_fn,
    length_truncated_stream_fn,
)


def _model() -> Model:
    return Model(
        id="mock",
        name="mock",
        api="mock",
        provider="mock",
        baseUrl="http://mock",
        reasoning=False,
        maxTokens=4096,
        contextWindow=100_000,
    )


def _usage() -> Usage:
    return Usage(input=0, output=0, cacheRead=0, cacheWrite=0, cacheWrite1h=0, reasoning=0, totalTokens=0)


def _convert_to_llm(messages):
    return messages


async def _grep_execute(tool_call_id, args, signal, on_update):
    return AgentToolResult(content=[TextContent(text="file.py: 3 TODO")], details={})


def _make_grep_tool() -> AgentTool:
    return AgentTool(
        name="grep",
        description="grep",
        parameters={
            "type": "object",
            "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
            "required": ["pattern", "path"],
        },
        label="Grep",
        execute=_grep_execute,
    )


def _base_config(tool_execution="sequential"):
    return AgentLoopConfig(
        model=_model(),
        convertToLlm=_convert_to_llm,
        toolExecution=tool_execution,
    )


async def _run(prompts, context, config, stream_fn):
    stream = agent_loop(prompts, context, config, None, stream_fn)
    events = [e async for e in stream]
    messages = await stream.result()
    return events, messages


def _types(events):
    return [e.type for e in events]


# ---------------------------------------------------------------------------
# Acceptance: grep -> re-inject tool result -> final answer
# ---------------------------------------------------------------------------
async def test_grep_demo_minimal_loop():
    tool = _make_grep_tool()
    config = _base_config()
    context = AgentContext(
        systemPrompt="You are a coding agent.",
        messages=[UserMessage(role="user", content="find TODOs", timestamp=0)],
        tools=[tool],
    )

    events, messages = await _run([], context, config, grep_demo_stream_fn)

    assert "agent_start" in _types(events)
    assert "agent_end" in _types(events)
    assert any(e.type == "tool_execution_start" and e.toolName == "grep" for e in events)
    assert any(e.type == "tool_execution_end" and e.toolName == "grep" for e in events)

    # Turn 1 (tool call) + Turn 2 (final answer) => 2 assistant messages
    assistant_msgs = [m for m in messages if getattr(m, "role", None) == "assistant"]
    tool_results = [m for m in messages if getattr(m, "role", None) == "toolResult"]
    assert len(assistant_msgs) == 2
    assert assistant_msgs[0].stopReason == "toolUse"
    assert assistant_msgs[1].stopReason == "stop"
    assert len(tool_results) == 1
    assert tool_results[0].toolName == "grep"
    assert tool_results[0].content[0].text == "file.py: 3 TODO"


# ---------------------------------------------------------------------------
# Parallel tool dispatch
# ---------------------------------------------------------------------------
def _two_tool_calls_stream_fn(model, context, options=None):
    loop = asyncio.get_event_loop()
    stream = AssistantMessageEventStream(loop=loop)
    n_assistant = sum(1 for m in context.messages if getattr(m, "role", None) == "assistant")

    async def _produce():
        if n_assistant == 0:
            out = AssistantMessage(
                role="assistant",
                content=[
                    ToolCall(id="a", name="alpha", arguments={}),
                    ToolCall(id="b", name="beta", arguments={}),
                ],
                api="mock",
                provider="mock",
                model="mock",
                usage=_usage(),
                stopReason="toolUse",
                timestamp=0,
            )
            stream.push(AssistantMessageEvent(type="start", partial=out))
            stream.push(AssistantMessageEvent(type="toolcall_start", contentIndex=0, partial=out))
            stream.push(AssistantMessageEvent(type="toolcall_end", contentIndex=0, toolCall=out.content[0], partial=out))
            stream.push(AssistantMessageEvent(type="toolcall_start", contentIndex=1, partial=out))
            stream.push(AssistantMessageEvent(type="toolcall_end", contentIndex=1, toolCall=out.content[1], partial=out))
            stream.push(AssistantMessageEvent(type="done", reason="toolUse", partial=out))
        else:
            out = AssistantMessage(
                role="assistant",
                content=[TextContent(text="done")],
                api="mock",
                provider="mock",
                model="mock",
                usage=_usage(),
                stopReason="stop",
                timestamp=0,
            )
            stream.push(AssistantMessageEvent(type="start", partial=out))
            stream.push(AssistantMessageEvent(type="text_start", contentIndex=0, partial=out))
            stream.push(AssistantMessageEvent(type="text_end", contentIndex=0, content="done", partial=out))
            stream.push(AssistantMessageEvent(type="done", reason="stop", partial=out))
        stream.end()

    loop.create_task(_produce())
    return stream


_execution_order = []


async def _alpha_execute(tool_call_id, args, signal, on_update):
    _execution_order.append("alpha")
    await asyncio.sleep(0.02)
    return AgentToolResult(content=[TextContent(text="A")], details={})


async def _beta_execute(tool_call_id, args, signal, on_update):
    _execution_order.append("beta")
    return AgentToolResult(content=[TextContent(text="B")], details={})


async def test_parallel_tool_dispatch():
    _execution_order.clear()
    config = _base_config(tool_execution="parallel")
    context = AgentContext(
        systemPrompt="",
        messages=[UserMessage(role="user", content="run both", timestamp=0)],
        tools=[
            AgentTool(name="alpha", description="", parameters={}, label="", execute=_alpha_execute),
            AgentTool(name="beta", description="", parameters={}, label="", execute=_beta_execute),
        ],
    )
    events, messages = await _run([], context, config, _two_tool_calls_stream_fn)

    starts = [e.toolName for e in events if e.type == "tool_execution_start"]
    ends = [e.toolName for e in events if e.type == "tool_execution_end"]
    assert set(starts) == {"alpha", "beta"}
    assert set(ends) == {"alpha", "beta"}
    # Both tools executed.
    assert set(_execution_order) == {"alpha", "beta"}


# ---------------------------------------------------------------------------
# Sequential tool dispatch preserves order
# ---------------------------------------------------------------------------
async def test_sequential_tool_dispatch():
    _execution_order.clear()
    config = _base_config(tool_execution="sequential")
    context = AgentContext(
        systemPrompt="",
        messages=[UserMessage(role="user", content="run both", timestamp=0)],
        tools=[
            AgentTool(name="alpha", description="", parameters={}, label="", execute=_alpha_execute),
            AgentTool(name="beta", description="", parameters={}, label="", execute=_beta_execute),
        ],
    )
    events, messages = await _run([], context, config, _two_tool_calls_stream_fn)
    assert _execution_order == ["alpha", "beta"]


# ---------------------------------------------------------------------------
# length self-heal: truncated tool calls are failed, loop continues
# ---------------------------------------------------------------------------
async def test_length_self_heal():
    tool = _make_grep_tool()
    config = _base_config()
    context = AgentContext(
        systemPrompt="",
        messages=[UserMessage(role="user", content="go", timestamp=0)],
        tools=[tool],
    )
    events, messages = await _run([], context, config, length_truncated_stream_fn)

    # The truncated tool call must be reported as an error tool result.
    tool_results = [m for m in messages if getattr(m, "role", None) == "toolResult"]
    assert tool_results, "expected a tool result from the truncated call"
    assert tool_results[0].isError is True
    # The agent should still have produced a final (stop) turn.
    final = [m for m in messages if getattr(m, "role", None) == "assistant" and m.stopReason == "stop"]
    assert final, "expected the loop to continue to a final stop turn"


# ---------------------------------------------------------------------------
# error stop reason short-circuits
# ---------------------------------------------------------------------------
async def test_error_stop_reason():
    config = _base_config()
    context = AgentContext(
        systemPrompt="",
        messages=[UserMessage(role="user", content="go", timestamp=0)],
        tools=[],
    )
    events, messages = await _run([], context, config, error_stream_fn)

    assert any(e.type == "agent_end" for e in events)
    # No tool execution should have happened.
    assert not any(e.type == "tool_execution_start" for e in events)
    assistant = [m for m in messages if getattr(m, "role", None) == "assistant"]
    assert assistant and assistant[0].stopReason == "error"
