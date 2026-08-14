"""Anthropic Messages adapter (Python port of packages/ai/src/api/anthropic-messages.ts).

Ports the production side of the streaming call:

  * SSE decode -> RawMessageStreamEvent (``iterate_sse_messages`` + parse)
  * event mapping loop: message_start / content_block_* / message_delta ->
    AssistantMessageEventStream pushes (start, text_*, thinking_*, toolcall_*, done/error)
  * ``map_stop_reason`` and ``parse_streaming_json`` (4-layer fallback)
  * ``build_params`` + httpx transport (core path; OAuth/Copilot/deferred-tool
    edge cases are intentionally simplified and noted in code comments).

The consumption side (``for await (const event of response)``) lives in
pi-agent-core's ``agent_loop``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from ..event_stream import AssistantMessageEventStream
from ..models import calculate_cost
from ..options import AnthropicOptions, SimpleStreamOptions, StreamOptions
from ..types import (
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    ImageContent,
    Model,
    TextContent,
    ThinkingContent,
    Tool,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from ..utils.json_parse import parse_json_with_repair, parse_streaming_json
from ..utils.text import sanitize_surrogates
from .sse import iterate_sse_messages

ANTHROPIC_MESSAGE_EVENTS = {
    "message_start",
    "message_delta",
    "message_stop",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
}

FINE_GRAINED_TOOL_STREAMING_BETA = "fine-grained-tool-streaming-2025-05-14"
INTERLEAVED_THINKING_BETA = "interleaved-thinking-2025-05-14"


@dataclass
class StopReasonResult:
    stopReason: str
    errorMessage: Optional[str] = None


def map_stop_reason(reason: str, stop_details: Optional[Dict[str, Any]] = None) -> StopReasonResult:
    """Maps an Anthropic stop_reason to pi's StopReason (mirrors upstream)."""
    explanation = (stop_details or {}).get("explanation") if stop_details else None
    if reason == "end_turn":
        return StopReasonResult("stop")
    if reason == "max_tokens":
        return StopReasonResult("length")
    if reason == "tool_use":
        return StopReasonResult("toolUse")
    if reason == "refusal":
        return StopReasonResult("error", explanation or "The model refused to complete the request")
    if reason == "pause_turn":  # Resubmit
        return StopReasonResult("stop")
    if reason == "stop_sequence":
        return StopReasonResult("stop")
    if reason == "sensitive":  # Flagged by safety filters
        return StopReasonResult("error", "Provider stopped with: sensitive")
    raise ValueError(f"Unhandled stop reason: {reason}")


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
def _has_header(headers: Optional[Dict[str, Optional[str]]], name: str) -> bool:
    if not headers:
        return False
    expected = name.lower()
    for key, value in headers.items():
        if key.lower() == expected and value not in (None, ""):
            return True
    return False


def assert_request_auth(provider: str, api_key: Optional[str], headers: Optional[Dict[str, Optional[str]]]) -> None:
    if api_key:
        return
    if _has_header(headers, "authorization") or _has_header(headers, "x-api-key"):
        return
    raise ValueError(f"No API key for provider: {provider}")


# ---------------------------------------------------------------------------
# Content / message conversion (outgoing -> Anthropic API shape)
# ---------------------------------------------------------------------------
def convert_content_blocks(content: List[Any]) -> Any:
    """Assistant content blocks -> Anthropic content (string or block array)."""
    has_images = any(getattr(c, "type", None) == "image" for c in content)
    if not has_images:
        return sanitize_surrogates("\n".join(getattr(c, "text", "") for c in content))
    blocks = []
    for block in content:
        if getattr(block, "type", None) == "text":
            blocks.append({"type": "text", "text": sanitize_surrogates(block.text)})
        else:
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": block.mimeType,
                        "data": block.data,
                    },
                }
            )
    if not any(b["type"] == "text" for b in blocks):
        blocks.insert(0, {"type": "text", "text": "(see attached image)"})
    return blocks


def convert_messages(
    messages: List[Any],
    cache_control: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    params: List[Dict[str, Any]] = []
    for msg in messages:
        role = getattr(msg, "role", None)
        if role == "user":
            content = msg.content
            if isinstance(content, str):
                if content.strip():
                    params.append({"role": "user", "content": sanitize_surrogates(content)})
            else:
                blocks = []
                for item in content:
                    if getattr(item, "type", None) == "text":
                        blocks.append({"type": "text", "text": sanitize_surrogates(item.text)})
                    else:
                        blocks.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": item.mimeType,
                                    "data": item.data,
                                },
                            }
                        )
                blocks = [b for b in blocks if not (b["type"] == "text" and not b["text"].strip())]
                if blocks:
                    params.append({"role": "user", "content": blocks})
        elif role == "assistant":
            blocks = []
            for block in msg.content:
                btype = getattr(block, "type", None)
                if btype == "text":
                    if block.text.strip():
                        blocks.append({"type": "text", "text": sanitize_surrogates(block.text)})
                elif btype == "thinking":
                    if block.redacted:
                        blocks.append({"type": "redacted_thinking", "data": block.thinkingSignature})
                        continue
                    if block.thinkingSignature:
                        blocks.append(
                            {
                                "type": "thinking",
                                "thinking": sanitize_surrogates(block.thinking),
                                "signature": block.thinkingSignature,
                            }
                        )
                    else:
                        blocks.append({"type": "text", "text": sanitize_surrogates(block.thinking)})
                elif btype == "toolCall":
                    blocks.append(
                        {"type": "tool_use", "id": block.id, "name": block.name, "input": block.arguments or {}}
                    )
            if blocks:
                params.append({"role": "assistant", "content": blocks})
        elif role == "toolResult":
            params.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.toolCallId,
                            "content": convert_content_blocks(msg.content)
                            if isinstance(msg.content, list)
                            else sanitize_surrogates(msg.content),
                            "is_error": msg.isError,
                        }
                    ],
                }
            )

    if cache_control and params:
        last = params[-1]
        if last["role"] == "user":
            content = last["content"]
            if isinstance(content, list):
                last_block = content[-1]
                if last_block["type"] in ("text", "image", "tool_result"):
                    last_block["cache_control"] = cache_control
            else:
                last["content"] = [{"type": "text", "text": content, "cache_control": cache_control}]
    return params


def convert_tools(tools: List[Tool], cache_control: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    out = []
    for i, tool in enumerate(tools):
        schema = tool.parameters or {}
        input_schema = {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", []),
        }
        tool_def: Dict[str, Any] = {
            "name": tool.name,
            "description": tool.description,
            "input_schema": input_schema,
        }
        if cache_control and i == len(tools) - 1:
            tool_def["cache_control"] = cache_control
        out.append(tool_def)
    return out


def resolve_cache_retention(options: Optional[StreamOptions]) -> str:
    if options and getattr(options, "cacheRetention", None):
        return options.cacheRetention  # type: ignore[attr-defined]
    return "short"


def build_params(model: Model, context: Context, options: Optional[StreamOptions]) -> Dict[str, Any]:
    """Build the Anthropic ``messages.create`` request body (core path).

    Simplified vs upstream: OAuth/Copilot auth, deferred-tool references, and
    per-provider compat flags are omitted; the essential fields
    (model, messages, system, max_tokens, tools, temperature, thinking) are kept.
    """
    from ..types import AssistantMessage  # noqa: F401  (kept for parity with upstream import)

    cache_retention = resolve_cache_retention(options)
    cache_control = (
        {"type": "ephemeral"}
        if cache_retention in ("short", "long")
        else None
    )

    params: Dict[str, Any] = {
        "model": model.id,
        "messages": convert_messages(context.messages, cache_control),
        "max_tokens": (options.maxTokens if options and options.maxTokens else model.maxTokens),
        "stream": True,
    }

    if context.systemPrompt:
        sys_block = {"type": "text", "text": sanitize_surrogates(context.systemPrompt)}
        if cache_control:
            sys_block["cache_control"] = cache_control
        params["system"] = [sys_block]

    opts = options or StreamOptions()
    thinking_enabled = getattr(opts, "thinkingEnabled", None)
    if opts.temperature is not None and not thinking_enabled:
        params["temperature"] = opts.temperature

    if context.tools:
        params["tools"] = convert_tools(context.tools, cache_control)

    if getattr(model, "reasoning", False):
        if thinking_enabled:
            display = getattr(opts, "thinkingDisplay", "summarized") or "summarized"
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": getattr(opts, "thinkingBudgetTokens", None) or 1024,
                "display": display,
            }
        elif thinking_enabled is False:
            params["thinking"] = {"type": "disabled"}

    if opts.toolChoice:
        if isinstance(opts.toolChoice, str):
            params["tool_choice"] = {"type": opts.toolChoice}
        else:
            params["tool_choice"] = opts.toolChoice

    return params


# ---------------------------------------------------------------------------
# Stream implementation
# ---------------------------------------------------------------------------
def stream(
    model: Model,
    context: Context,
    options: Optional[AnthropicOptions] = None,
) -> AssistantMessageEventStream:
    """Stream an Anthropic Messages response, emitting AssistantMessageEvents."""
    loop = asyncio.get_event_loop()
    stream = AssistantMessageEventStream(loop=loop)

    loop.create_task(_run_stream(model, context, options, stream))
    return stream


async def _run_stream(
    model: Model,
    context: Context,
    options: Optional[AnthropicOptions],
    stream: AssistantMessageEventStream,
) -> None:
    output = AssistantMessage(
        role="assistant",
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(
            input=0,
            output=0,
            cacheRead=0,
            cacheWrite=0,
            cacheWrite1h=0,
            reasoning=0,
            totalTokens=0,
        ),
        stopReason="pending",
        timestamp=0,
    )

    try:
        api_key = options.apiKey if options else None
        headers = options.headers if options else None
        assert_request_auth(model.provider, api_key, headers)

        abort_event = options.signal if options else None
        params = build_params(model, context, options)

        request_headers: Dict[str, str] = {
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        if api_key:
            request_headers["x-api-key"] = api_key
        if headers:
            for k, v in headers.items():
                if v is not None:
                    request_headers[k] = v
        beta: List[str] = []
        if context.tools:
            beta.append(FINE_GRAINED_TOOL_STREAMING_BETA)
        if getattr(options, "interleavedThinking", True) and getattr(model, "reasoning", False):
            beta.append(INTERLEAVED_THINKING_BETA)
        if beta:
            request_headers["anthropic-beta"] = ",".join(beta)

        url = model.baseUrl.rstrip("/") + "/messages"
        timeout = (options.timeout / 1000.0) if options and options.timeout else 600.0

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST", url, headers=request_headers, json=params, timeout=timeout
            ) as response:
                if options and options.onResponse:
                    await _maybe_await(
                        options.onResponse(
                            {"status": response.status_code, "headers": dict(response.headers)}, model
                        )
                    )
                stream.push(AssistantMessageEvent(type="start", partial=output))

                blocks = output.content
                saw_start = False
                saw_end = False

                async for sse in iterate_sse_messages(response.aiter_lines(), abort_event):
                    if sse.event == "error":
                        raise RuntimeError(sse.data)
                    if sse.event not in ANTHROPIC_MESSAGE_EVENTS:
                        continue
                    try:
                        event = parse_json_with_repair(sse.data)
                    except Exception as e:
                        raise RuntimeError(
                            f"Could not parse Anthropic SSE event {sse.event}: {e}; data={sse.data}"
                        ) from e

                    etype = event.get("type")
                    if etype == "message_start":
                        saw_start = True
                        msg = event.get("message", {})
                        usage = msg.get("usage", {})
                        output.responseId = msg.get("id")
                        output.usage.input = usage.get("input_tokens", 0) or 0
                        output.usage.output = usage.get("output_tokens", 0) or 0
                        output.usage.cacheRead = usage.get("cache_read_input_tokens", 0) or 0
                        output.usage.cacheWrite = usage.get("cache_creation_input_tokens", 0) or 0
                        output.usage.cacheWrite1h = usage.get("cache_creation", {}).get(
                            "ephemeral_1h_input_tokens", 0
                        ) or 0
                        output.usage.totalTokens = (
                            output.usage.input
                            + output.usage.output
                            + output.usage.cacheRead
                            + output.usage.cacheWrite
                        )
                        calculate_cost(model, output.usage)
                    elif etype == "content_block_start":
                        cb = event.get("content_block", {})
                        cbtype = cb.get("type")
                        block: Any
                        if cbtype == "text":
                            block = TextContent(text=cb.get("text") or "")
                            block.index = event["index"]  # type: ignore[attr-defined]
                            output.content.append(block)
                            stream.push(
                                AssistantMessageEvent(
                                    type="text_start", contentIndex=len(output.content) - 1, partial=output
                                )
                            )
                        elif cbtype == "thinking":
                            block = ThinkingContent(
                                thinking=cb.get("thinking") or "",
                                thinkingSignature=cb.get("signature") or "",
                            )
                            block.index = event["index"]  # type: ignore[attr-defined]
                            output.content.append(block)
                            stream.push(
                                AssistantMessageEvent(
                                    type="thinking_start", contentIndex=len(output.content) - 1, partial=output
                                )
                            )
                        elif cbtype == "tool_use":
                            block = ToolCall(
                                id=cb.get("id", ""),
                                name=cb.get("name", ""),
                                arguments=cb.get("input") or {},
                            )
                            block.index = event["index"]  # type: ignore[attr-defined]
                            block.partialJson = ""  # type: ignore[attr-defined]
                            output.content.append(block)
                            stream.push(
                                AssistantMessageEvent(
                                    type="toolcall_start", contentIndex=len(output.content) - 1, partial=output
                                )
                            )
                    elif etype == "content_block_delta":
                        delta = event.get("delta", {})
                        dtype = delta.get("type")
                        idx = event["index"]
                        block = _find_block(blocks, idx)
                        if dtype == "text_delta" and block is not None and getattr(block, "type", None) == "text":
                            block.text += delta.get("text", "")
                            stream.push(
                                AssistantMessageEvent(
                                    type="text_delta",
                                    contentIndex=idx,
                                    delta=delta.get("text", ""),
                                    partial=output,
                                )
                            )
                        elif dtype == "thinking_delta" and block is not None and getattr(block, "type", None) == "thinking":
                            block.thinking += delta.get("thinking", "")
                            stream.push(
                                AssistantMessageEvent(
                                    type="thinking_delta",
                                    contentIndex=idx,
                                    delta=delta.get("thinking", ""),
                                    partial=output,
                                )
                            )
                        elif dtype == "input_json_delta" and block is not None and getattr(block, "type", None) == "toolCall":
                            block.partialJson += delta.get("partial_json", "")  # type: ignore[attr-defined]
                            block.arguments = parse_streaming_json(block.partialJson)  # type: ignore[attr-defined]
                            stream.push(
                                AssistantMessageEvent(
                                    type="toolcall_delta",
                                    contentIndex=idx,
                                    delta=delta.get("partial_json", ""),
                                    partial=output,
                                )
                            )
                        elif dtype == "signature_delta" and block is not None and getattr(block, "type", None) == "thinking":
                            block.thinkingSignature = (block.thinkingSignature or "") + delta.get("signature", "")
                    elif etype == "content_block_stop":
                        idx = event["index"]
                        block = _find_block(blocks, idx)
                        if block is not None:
                            del block.index  # type: ignore[attr-defined]
                            btype = getattr(block, "type", None)
                            if btype == "text":
                                stream.push(
                                    AssistantMessageEvent(
                                        type="text_end",
                                        contentIndex=idx,
                                        content=block.text,
                                        partial=output,
                                    )
                                )
                            elif btype == "thinking":
                                stream.push(
                                    AssistantMessageEvent(
                                        type="thinking_end",
                                        contentIndex=idx,
                                        content=block.thinking,
                                        partial=output,
                                    )
                                )
                            elif btype == "toolCall":
                                block.arguments = parse_streaming_json(block.partialJson)  # type: ignore[attr-defined]
                                del block.partialJson  # type: ignore[attr-defined]
                                stream.push(
                                    AssistantMessageEvent(
                                        type="toolcall_end",
                                        contentIndex=idx,
                                        toolCall=block,
                                        partial=output,
                                    )
                                )
                    elif etype == "message_delta":
                        delta = event.get("delta", {})
                        if delta.get("stop_reason"):
                            output.rawStopReason = delta.get("stop_reason")
                            res = map_stop_reason(delta.get("stop_reason"), delta.get("stop_details"))
                            output.stopReason = res.stopReason  # type: ignore[assignment]
                            if res.errorMessage:
                                output.errorMessage = res.errorMessage
                        usage = event.get("usage")
                        if usage:
                            if usage.get("input_tokens") is not None:
                                output.usage.input = usage["input_tokens"]
                            if usage.get("output_tokens") is not None:
                                output.usage.output = usage["output_tokens"]
                            if usage.get("cache_read_input_tokens") is not None:
                                output.usage.cacheRead = usage["cache_read_input_tokens"]
                            if usage.get("cache_creation_input_tokens") is not None:
                                output.usage.cacheWrite = usage["cache_creation_input_tokens"]
                            thinking_tokens = (usage.get("output_tokens_details") or {}).get("thinking_tokens")
                            if thinking_tokens is not None:
                                output.usage.reasoning = thinking_tokens
                        output.usage.totalTokens = (
                            output.usage.input
                            + output.usage.output
                            + output.usage.cacheRead
                            + output.usage.cacheWrite
                        )
                        calculate_cost(model, output.usage)

                if abort_event is not None and abort_event.is_set():
                    raise asyncio.CancelledError("Request was aborted")
                if output.stopReason == "pending":
                    raise RuntimeError("Anthropic stream ended without a stop reason")
                if output.stopReason in ("aborted", "error"):
                    raise RuntimeError(output.errorMessage or "An unknown error occurred")

                stream.push(
                    AssistantMessageEvent(type="done", reason=output.stopReason, partial=output)
                )
                stream.end()
    except Exception as e:
        for block in output.content:
            if hasattr(block, "index"):
                del block.index  # type: ignore[attr-defined]
            if hasattr(block, "partialJson"):
                del block.partialJson  # type: ignore[attr-defined]
        aborted = abort_event is not None and abort_event.is_set()
        output.stopReason = "aborted" if aborted else "error"  # type: ignore[assignment]
        output.errorMessage = str(e)
        stream.push(
            AssistantMessageEvent(type="error", reason=output.stopReason, error=output)
        )
        stream.end()


def _find_block(blocks: List[Any], index: int) -> Any:
    for b in blocks:
        if getattr(b, "index", None) == index:
            return b
    return None


async def _maybe_await(value: Any) -> None:
    if asyncio.iscoroutine(value):
        await value


def stream_simple(
    model: Model,
    context: Context,
    options: Optional[SimpleStreamOptions] = None,
) -> AssistantMessageEventStream:
    """Simplified stream entry point (mirrors upstream streamSimple)."""
    assert_request_auth(model.provider, options.apiKey if options else None, options.headers if options else None)

    if options is None or not options.reasoning:
        return stream(
            model,
            context,
            AnthropicOptions(apiKey=options.apiKey if options else None, thinkingEnabled=False)
            if options is None
            else AnthropicOptions(
                apiKey=options.apiKey,
                headers=options.headers,
                maxTokens=options.maxTokens,
                temperature=options.temperature,
                cacheRetention=options.cacheRetention,
                sessionId=options.sessionId,
                signal=options.signal,
                metadata=options.metadata,
                toolChoice=options.toolChoice,
                env=options.env,
                onResponse=options.onResponse,
                onPayload=options.onPayload,
                thinkingEnabled=False,
            ),
        )

    base_max = options.maxTokens if options.maxTokens else model.maxTokens
    budget = max(0, min(base_max - 1024, 1024))
    return stream(
        model,
        context,
        AnthropicOptions(
            apiKey=options.apiKey,
            headers=options.headers,
            maxTokens=base_max,
            temperature=options.temperature,
            cacheRetention=options.cacheRetention,
            sessionId=options.sessionId,
            signal=options.signal,
            metadata=options.metadata,
            toolChoice=options.toolChoice,
            env=options.env,
            onResponse=options.onResponse,
            onPayload=options.onPayload,
            thinkingEnabled=True,
            thinkingBudgetTokens=budget,
        ),
    )


__all__ = ["stream", "stream_simple", "map_stop_reason", "build_params", "convert_messages", "convert_tools"]
