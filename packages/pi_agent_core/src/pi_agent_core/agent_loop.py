"""Agent loop (Python port of packages/agent/src/agent-loop.ts).

Works with AgentMessage throughout; transforms to Message[] only at the LLM
call boundary (``convertToLlm``). Mirrors the upstream double-loop:

    runLoop (outer): drains follow-up messages after the agent would stop
      inner:      process tool calls + steering messages until no more

Event protocol is identical: agent_start / turn_start / message_start /
message_update / message_end / tool_execution_* / turn_end / agent_end.
"""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field
from typing import Any, List, Optional

from pi_ai import (
    AssistantMessage,
    AssistantMessageEvent,
    AssistantMessageEventStream,
    Context,
    EventStream,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    validate_tool_arguments,
)

from .stream_fn import get_default_stream_fn
from .types import (
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    AgentTool,
    AgentToolCall,
    AgentToolResult,
    AfterToolCallContext,
    BeforeToolCallContext,
)


def agent_loop(
    prompts: List[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    signal: Optional[asyncio.Event],
    stream_fn,
) -> EventStream[AgentEvent, List[AgentMessage]]:
    stream = _create_agent_stream()
    loop = asyncio.get_event_loop()

    async def _emit(event: AgentEvent) -> None:
        # The upstream emit is a void (synchronous) sink; push() is sync,
        # but callers await emit(), so wrap it as a coroutine.
        stream.push(event)

    async def _drive() -> None:
        try:
            messages = await run_agent_loop(prompts, context, config, _emit, signal, stream_fn)
            if not stream._done:  # type: ignore[attr-defined]
                stream.end(messages)
        except Exception:
            if not stream._done:  # type: ignore[attr-defined]
                stream.end([])

    loop.create_task(_drive())
    return stream


def agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: Optional[asyncio.Event],
    stream_fn,
) -> EventStream[AgentEvent, List[AgentMessage]]:
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")
    if getattr(context.messages[-1], "role", None) == "assistant":
        raise ValueError("Cannot continue from message role: assistant")

    stream = _create_agent_stream()
    loop = asyncio.get_event_loop()

    async def _emit(event: AgentEvent) -> None:
        stream.push(event)

    async def _drive() -> None:
        try:
            messages = await run_agent_loop_continue(context, config, _emit, signal, stream_fn)
            if not stream._done:  # type: ignore[attr-defined]
                stream.end(messages)
        except Exception:
            if not stream._done:  # type: ignore[attr-defined]
                stream.end([])

    loop.create_task(_drive())
    return stream


async def run_agent_loop(
    prompts: List[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    emit,
    signal: Optional[asyncio.Event],
    stream_fn,
) -> List[AgentMessage]:
    new_messages: List[AgentMessage] = list(prompts)
    current_context = AgentContext(
        systemPrompt=context.systemPrompt,
        messages=[*context.messages, *prompts],
        tools=context.tools,
    )

    await emit(AgentEvent(type="agent_start"))
    await emit(AgentEvent(type="turn_start"))
    for prompt in prompts:
        await emit(AgentEvent(type="message_start", message=prompt))
        await emit(AgentEvent(type="message_end", message=prompt))

    await run_loop(current_context, new_messages, config, signal, emit, stream_fn)
    return new_messages


async def run_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    emit,
    signal: Optional[asyncio.Event],
    stream_fn,
) -> List[AgentMessage]:
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")
    if getattr(context.messages[-1], "role", None) == "assistant":
        raise ValueError("Cannot continue from message role: assistant")

    new_messages: List[AgentMessage] = []
    current_context = AgentContext(
        systemPrompt=context.systemPrompt, messages=list(context.messages), tools=context.tools
    )

    await emit(AgentEvent(type="agent_start"))
    await emit(AgentEvent(type="turn_start"))

    await run_loop(current_context, new_messages, config, signal, emit, stream_fn)
    return new_messages


def _create_agent_stream() -> EventStream[AgentEvent, List[AgentMessage]]:
    return EventStream[AgentEvent, List[AgentMessage]](
        is_complete=lambda e: e.type == "agent_end",
        extract_result=lambda e: e.messages if e.messages is not None else [],
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
async def run_loop(
    initial_context: AgentContext,
    new_messages: List[AgentMessage],
    initial_config: AgentLoopConfig,
    signal: Optional[asyncio.Event],
    emit,
    stream_function,
) -> None:
    current_context = initial_context
    config = initial_config
    first_turn = True
    pending_messages: List[AgentMessage] = await _maybe(config.getSteeringMessages()) if config.getSteeringMessages else []

    while True:
        has_more_tool_calls = True

        while has_more_tool_calls or pending_messages:
            if not first_turn:
                await emit(AgentEvent(type="turn_start"))
            else:
                first_turn = False

            if pending_messages:
                for message in pending_messages:
                    await emit(AgentEvent(type="message_start", message=message))
                    await emit(AgentEvent(type="message_end", message=message))
                    current_context.messages.append(message)
                    new_messages.append(message)
                pending_messages = []

            message = await stream_assistant_response(current_context, config, signal, emit, stream_function)
            new_messages.append(message)

            if message.stopReason in ("error", "aborted"):
                await emit(AgentEvent(type="turn_end", message=message, toolResults=[]))
                await emit(AgentEvent(type="agent_end", messages=list(new_messages)))
                return

            tool_calls = [c for c in message.content if getattr(c, "type", None) == "toolCall"]
            tool_results: List[ToolResultMessage] = []
            has_more_tool_calls = False
            if tool_calls:
                executed = (
                    await fail_tool_calls_from_truncated_message(tool_calls, emit)
                    if message.stopReason == "length"
                    else await execute_tool_calls(current_context, message, config, signal, emit)
                )
                tool_results.extend(executed.messages)
                has_more_tool_calls = not executed.terminate

                for result in tool_results:
                    current_context.messages.append(result)
                    new_messages.append(result)

            await emit(AgentEvent(type="turn_end", message=message, toolResults=tool_results))

            next_turn_context = {
                "message": message,
                "toolResults": tool_results,
                "context": current_context,
                "newMessages": new_messages,
            }
            next_turn_snapshot = await _maybe(config.prepareNextTurn(next_turn_context)) if config.prepareNextTurn else None
            if next_turn_snapshot:
                current_context = next_turn_snapshot.context or current_context
                config = _apply_turn_update(config, next_turn_snapshot)

            if config.shouldStopAfterTurn:
                should_stop = await _maybe(
                    config.shouldStopAfterTurn(
                        _ShouldStopAfterTurnContext(message, tool_results, current_context, new_messages)
                    )
                )
                if should_stop:
                    await emit(AgentEvent(type="agent_end", messages=list(new_messages)))
                    return

            pending_messages = await _maybe(config.getSteeringMessages()) if config.getSteeringMessages else []

        follow_up = await _maybe(config.getFollowUpMessages()) if config.getFollowUpMessages else []
        if follow_up:
            pending_messages = follow_up
            continue
        break

    await emit(AgentEvent(type="agent_end", messages=list(new_messages)))


def _apply_turn_update(config: AgentLoopConfig, snapshot):
    if snapshot.model is not None:
        config = _replace(config, model=snapshot.model)
    if snapshot.thinkingLevel is not None:
        config = _replace(
            config,
            thinkingEnabled=None if snapshot.thinkingLevel == "off" else True,
            reasoning=snapshot.thinkingLevel if snapshot.thinkingLevel != "off" else None,
        )
    return config


# ---------------------------------------------------------------------------
# Streaming one assistant response (consumption side)
# ---------------------------------------------------------------------------
async def stream_assistant_response(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: Optional[asyncio.Event],
    emit,
    stream_function,
) -> AssistantMessage:
    messages = context.messages
    if config.transformContext:
        messages = await _maybe(config.transformContext(messages, signal))

    llm_messages = await _maybe(config.convertToLlm(messages))

    llm_context = Context(
        systemPrompt=context.systemPrompt,
        messages=llm_messages,
        tools=context.tools,
    )

    resolved_api_key = (
        (await _maybe(config.getApiKey(config.model.provider)) if config.getApiKey else None)
        or config.apiKey
    )
    options = _replace(config, apiKey=resolved_api_key, signal=signal)
    response = await _maybe(stream_function(config.model, llm_context, options))

    partial_message: Optional[AssistantMessage] = None
    added_partial = False

    async for event in response:
        if event.type == "start":
            partial_message = event.partial
            context.messages.append(partial_message)
            added_partial = True
            await emit(AgentEvent(type="message_start", message=_clone(partial_message)))
        elif event.type in (
            "text_start",
            "text_delta",
            "text_end",
            "thinking_start",
            "thinking_delta",
            "thinking_end",
            "toolcall_start",
            "toolcall_delta",
            "toolcall_end",
        ):
            if partial_message is not None:
                partial_message = event.partial
                context.messages[-1] = partial_message
                await emit(
                    AgentEvent(
                        type="message_update",
                        assistantMessageEvent=event,
                        message=_clone(partial_message),
                    )
                )
        elif event.type in ("done", "error"):
            final_message = await response.result()
            if added_partial:
                context.messages[-1] = final_message
            else:
                context.messages.append(final_message)
            if not added_partial:
                await emit(AgentEvent(type="message_start", message=_clone(final_message)))
            await emit(AgentEvent(type="message_end", message=final_message))
            return final_message

    final_message = await response.result()
    if added_partial:
        context.messages[-1] = final_message
    else:
        context.messages.append(final_message)
        await emit(AgentEvent(type="message_start", message=_clone(final_message)))
    await emit(AgentEvent(type="message_end", message=final_message))
    return final_message


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------
@dataclass
class _ExecutedBatch:
    messages: List[ToolResultMessage]
    terminate: bool


async def fail_tool_calls_from_truncated_message(
    tool_calls: List[AgentToolCall], emit
) -> _ExecutedBatch:
    messages: List[ToolResultMessage] = []
    for tool_call in tool_calls:
        await emit(
            AgentEvent(
                type="tool_execution_start",
                toolCallId=tool_call.id,
                toolName=tool_call.name,
                args=tool_call.arguments,
            )
        )
        finalized = {
            "toolCall": tool_call,
            "result": _create_error_tool_result(
                f'Tool call "{tool_call.name}" was not executed: the response hit the output token limit, '
                "so its arguments may be truncated. Re-issue the tool call with complete arguments."
            ),
            "isError": True,
        }
        await _emit_tool_execution_end(finalized, emit)
        tool_result_message = _create_tool_result_message(finalized)
        await _emit_tool_result_message(tool_result_message, emit)
        messages.append(tool_result_message)
    return _ExecutedBatch(messages=messages, terminate=False)


async def execute_tool_calls(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    config: AgentLoopConfig,
    signal: Optional[asyncio.Event],
    emit,
) -> _ExecutedBatch:
    tool_calls = [c for c in assistant_message.content if getattr(c, "type", None) == "toolCall"]
    has_sequential = any(
        current_context.tools
        and any(t.name == tc.name and getattr(t, "executionMode", None) == "sequential" for t in current_context.tools)
        for tc in tool_calls
    )
    if config.toolExecution == "sequential" or has_sequential:
        return await execute_tool_calls_sequential(current_context, assistant_message, tool_calls, config, signal, emit)
    return await execute_tool_calls_parallel(current_context, assistant_message, tool_calls, config, signal, emit)


async def execute_tool_calls_sequential(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: List[AgentToolCall],
    config: AgentLoopConfig,
    signal: Optional[asyncio.Event],
    emit,
) -> _ExecutedBatch:
    finalized_calls: List[dict] = []
    messages: List[ToolResultMessage] = []

    for tool_call in tool_calls:
        await emit(
            AgentEvent(
                type="tool_execution_start",
                toolCallId=tool_call.id,
                toolName=tool_call.name,
                args=tool_call.arguments,
            )
        )
        preparation = await prepare_tool_call(current_context, assistant_message, tool_call, config, signal)
        if preparation["kind"] == "immediate":
            finalized = {
                "toolCall": tool_call,
                "result": preparation["result"],
                "isError": preparation["isError"],
            }
        else:
            executed = await execute_prepared_tool_call(preparation, signal, emit)
            finalized = await finalize_executed_tool_call(
                current_context, assistant_message, preparation, executed, config, signal
            )
        await _emit_tool_execution_end(finalized, emit)
        tool_result_message = _create_tool_result_message(finalized)
        await _emit_tool_result_message(tool_result_message, emit)
        finalized_calls.append(finalized)
        messages.append(tool_result_message)
        if signal is not None and signal.is_set():
            break

    return _ExecutedBatch(messages=messages, terminate=_should_terminate_tool_batch(finalized_calls))


async def execute_tool_calls_parallel(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: List[AgentToolCall],
    config: AgentLoopConfig,
    signal: Optional[asyncio.Event],
    emit,
) -> _ExecutedBatch:
    finalized_calls: List[Any] = []

    for tool_call in tool_calls:
        await emit(
            AgentEvent(
                type="tool_execution_start",
                toolCallId=tool_call.id,
                toolName=tool_call.name,
                args=tool_call.arguments,
            )
        )
        preparation = await prepare_tool_call(current_context, assistant_message, tool_call, config, signal)
        if preparation["kind"] == "immediate":
            finalized = {
                "toolCall": tool_call,
                "result": preparation["result"],
                "isError": preparation["isError"],
            }
            await _emit_tool_execution_end(finalized, emit)
            finalized_calls.append(finalized)
            if signal is not None and signal.is_set():
                break
            continue

        finalized_calls.append(
            _make_parallel_entry(preparation, current_context, assistant_message, config, signal, emit)
        )
        if signal is not None and signal.is_set():
            break

    ordered_finalized = []
    for entry in finalized_calls:
        if callable(entry):
            ordered_finalized.append(await entry())
        else:
            ordered_finalized.append(entry)

    messages: List[ToolResultMessage] = []
    for finalized in ordered_finalized:
        tool_result_message = _create_tool_result_message(finalized)
        await _emit_tool_result_message(tool_result_message, emit)
        messages.append(tool_result_message)

    return _ExecutedBatch(messages=messages, terminate=_should_terminate_tool_batch(ordered_finalized))


def _make_parallel_entry(preparation, current_context, assistant_message, config, signal, emit):
    async def entry():
        executed = await execute_prepared_tool_call(preparation, signal, emit)
        finalized = await finalize_executed_tool_call(
            current_context, assistant_message, preparation, executed, config, signal
        )
        await _emit_tool_execution_end(finalized, emit)
        return finalized

    return entry


def _should_terminate_tool_batch(finalized_calls: List[dict]) -> bool:
    return bool(finalized_calls) and all(f["result"].terminate for f in finalized_calls)


def prepare_tool_call_arguments(tool: AgentTool, tool_call: AgentToolCall) -> AgentToolCall:
    if not tool.prepareArguments:
        return tool_call
    prepared = tool.prepareArguments(tool_call.arguments)
    if prepared is tool_call.arguments:
        return tool_call
    return _replace(tool_call, arguments=prepared)


async def prepare_tool_call(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_call: AgentToolCall,
    config: AgentLoopConfig,
    signal: Optional[asyncio.Event],
):
    tool = (
        next((t for t in (current_context.tools or []) if t.name == tool_call.name), None)
        if current_context.tools
        else None
    )
    if tool is None:
        return {
            "kind": "immediate",
            "result": _create_error_tool_result(f"Tool {tool_call.name} not found"),
            "isError": True,
        }

    try:
        prepared_tool_call = prepare_tool_call_arguments(tool, tool_call)
        validated_args = validate_tool_arguments(tool, prepared_tool_call)
        if config.beforeToolCall:
            before_result = await _maybe(
                config.beforeToolCall(
                    _BeforeToolCallContext(assistant_message, tool_call, validated_args, current_context),
                    signal,
                )
            )
            if signal is not None and signal.is_set():
                return {
                    "kind": "immediate",
                    "result": _create_error_tool_result("Operation aborted"),
                    "isError": True,
                }
            if before_result and getattr(before_result, "block", False):
                result = _create_error_tool_result(before_result.reason or "Tool execution was blocked")
                if getattr(before_result, "terminate", False):
                    result.terminate = True
                return {"kind": "immediate", "result": result, "isError": True}
        if signal is not None and signal.is_set():
            return {
                "kind": "immediate",
                "result": _create_error_tool_result("Operation aborted"),
                "isError": True,
            }
        return {"kind": "prepared", "toolCall": tool_call, "tool": tool, "args": validated_args}
    except Exception as e:
        return {
            "kind": "immediate",
            "result": _create_error_tool_result(str(e) if isinstance(e, Exception) else str(e)),
            "isError": True,
        }


async def execute_prepared_tool_call(preparation, signal, emit) -> dict:
    update_events: List[asyncio.Future] = []
    accepting_updates = True

    def on_update(partial_result):
        if not accepting_updates:
            return
        update_events.append(
            asyncio.ensure_future(
                emit(
                    AgentEvent(
                        type="tool_execution_update",
                        toolCallId=preparation["toolCall"].id,
                        toolName=preparation["toolCall"].name,
                        args=preparation["toolCall"].arguments,
                        partialResult=partial_result,
                    )
                )
            )
        )

    try:
        result = await preparation["tool"].execute(
            preparation["toolCall"].id, preparation["args"], signal, on_update
        )
        accepting_updates = False
        await asyncio.gather(*update_events, return_exceptions=True)
        return {"result": result, "isError": False}
    except Exception as e:
        accepting_updates = False
        await asyncio.gather(*update_events, return_exceptions=True)
        return {"result": _create_error_tool_result(str(e)), "isError": True}
    finally:
        accepting_updates = False


async def finalize_executed_tool_call(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    prepared,
    executed: dict,
    config: AgentLoopConfig,
    signal: Optional[asyncio.Event],
) -> dict:
    result = executed["result"]
    is_error = executed["isError"]

    if config.afterToolCall:
        try:
            after_result = await _maybe(
                config.afterToolCall(
                    _AfterToolCallContext(
                        assistant_message,
                        prepared["toolCall"],
                        prepared["args"],
                        result,
                        is_error,
                        current_context,
                    ),
                    signal,
                )
            )
            if after_result:
                result = _replace(
                    result,
                    content=after_result.content if after_result.content is not None else result.content,
                    details=after_result.details if after_result.details is not None else result.details,
                    usage=after_result.usage if after_result.usage is not None else result.usage,
                    terminate=after_result.terminate if after_result.terminate is not None else result.terminate,
                )
                is_error = after_result.isError if after_result.isError is not None else is_error
        except Exception as e:
            result = _create_error_tool_result(str(e))
            is_error = True

    return {"toolCall": prepared["toolCall"], "result": result, "isError": is_error}


def _create_error_tool_result(message: str) -> AgentToolResult:
    return AgentToolResult(content=[TextContent(text=message)], details={})


async def _emit_tool_execution_end(finalized: dict, emit) -> None:
    await emit(
        AgentEvent(
            type="tool_execution_end",
            toolCallId=finalized["toolCall"].id,
            toolName=finalized["toolCall"].name,
            result=finalized["result"],
            isError=finalized["isError"],
        )
    )


def _create_tool_result_message(finalized: dict) -> ToolResultMessage:
    result: AgentToolResult = finalized["result"]
    return ToolResultMessage(
        role="toolResult",
        toolCallId=finalized["toolCall"].id,
        toolName=finalized["toolCall"].name,
        content=result.content if result.content is not None else [],
        details=result.details,
        usage=result.usage,
        addedToolNames=result.addedToolNames,
        isError=finalized["isError"],
        timestamp=0,
    )


async def _emit_tool_result_message(tool_result_message: ToolResultMessage, emit) -> None:
    await emit(AgentEvent(type="message_start", message=tool_result_message))
    await emit(AgentEvent(type="message_end", message=tool_result_message))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
async def _maybe(value):
    if asyncio.iscoroutine(value):
        return await value
    if isinstance(value, asyncio.Future):
        return await value
    return value


def _clone(msg: AssistantMessage) -> AssistantMessage:
    return copy.copy(msg)


def _replace(obj, **changes):
    return _dataclasses_replace(obj, **changes)


def _dataclasses_replace(obj, **changes):
    import dataclasses

    return dataclasses.replace(obj, **changes)


# Lightweight context dataclasses used above (kept local to avoid import cycles).
from dataclasses import dataclass as _dc  # noqa: E402


@_dc
class _ShouldStopAfterTurnContext:
    message: AssistantMessage
    toolResults: List[ToolResultMessage]
    context: AgentContext
    newMessages: List[AgentMessage]


@_dc
class _BeforeToolCallContext:
    assistantMessage: AssistantMessage
    toolCall: AgentToolCall
    args: Any
    context: AgentContext


@_dc
class _AfterToolCallContext:
    assistantMessage: AssistantMessage
    toolCall: AgentToolCall
    args: Any
    result: AgentToolResult
    isError: bool
    context: AgentContext


__all__ = [
    "agent_loop",
    "agent_loop_continue",
    "run_agent_loop",
    "run_agent_loop_continue",
]
