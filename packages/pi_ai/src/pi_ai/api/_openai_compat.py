"""OpenAI-Compatible Chat Completions adapter (Python port of packages/ai/src/api/openai-completions.ts).

This is the shared engine behind ``openai-completions``. The upstream routes
OpenRouter / Qwen / DeepSeek through this same module and only varies behavior
via a per-provider "compat" (mainly ``thinkingFormat``). We mirror that: a single
engine that auto-detects the provider from ``model.provider`` / ``model.baseUrl``
and emits the same ``AssistantMessageEvent`` sequence.

The HTTP transport is ``httpx`` directly (no ``openai`` SDK), keeping the port
Python-idiomatic and consistent with the Anthropic adapter. SSE line decoding is
decoupled from the network so the chunk-mapping logic is unit-testable with a
canned line source.

Pragmatic simplifications vs upstream (noted, behavior-equivalent for common
inference): grammar-constrained tools, encrypted ``reasoning_details``, session
affinity headers, prompt-cache keys, and per-provider cache-control are omitted.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx

from ..event_stream import AssistantMessageEventStream
from ..models import calculate_cost
from ..options import SimpleStreamOptions, StreamOptions
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
)
from ..utils.json_parse import parse_json_with_repair, parse_streaming_json
from ..utils.text import sanitize_surrogates

# Provider -> default base URL + auth env var (used when model.baseUrl is empty).
PROVIDER_DEFAULTS: Dict[str, Dict[str, str]] = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "auth_env": "OPENROUTER_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "auth_env": "DEEPSEEK_API_KEY",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "auth_env": "DASHSCOPE_API_KEY",
    },
}

# Fields that may carry reasoning content across OpenAI-compatible endpoints.
REASONING_FIELDS = ("reasoning_content", "reasoning", "reasoning_text")


@dataclass
class OpenAICompat:
    """Resolved compatibility settings (pragmatic subset of upstream)."""

    thinking_format: str = "openai"  # openai | openrouter | qwen | deepseek
    requires_reasoning_content_on_assistant: bool = False
    supports_usage_in_streaming: bool = True
    max_tokens_field: str = "max_tokens"


def detect_compat(model: Model) -> OpenAICompat:
    provider = (model.provider or "").lower()
    base_url = (model.baseUrl or "").lower()

    is_openrouter = provider == "openrouter" or "openrouter.ai" in base_url
    is_deepseek = provider == "deepseek" or "deepseek.com" in base_url
    is_qwen = provider.startswith("qwen") or "dashscope" in base_url or "aliyuncs" in base_url

    if is_openrouter:
        thinking_format = "openrouter"
    elif is_deepseek:
        thinking_format = "deepseek"
    elif is_qwen:
        thinking_format = "qwen"
    else:
        thinking_format = "openai"

    # DeepSeek (and DeepSeek-reasoner) expect an empty reasoning_content on
    # assistant turns when the model supports reasoning.
    requires_reasoning_content = is_deepseek

    return OpenAICompat(
        thinking_format=thinking_format,
        requires_reasoning_content_on_assistant=requires_reasoning_content,
    )


def map_stop_reason(reason: Optional[str]) -> Tuple[str, Optional[str]]:
    """Maps an OpenAI finish_reason to pi's StopReason (mirrors upstream)."""
    if reason is None:
        return "stop", None
    if reason in ("stop", "end"):
        return "stop", None
    if reason == "length":
        return "length", None
    if reason in ("function_call", "tool_calls"):
        return "toolUse", None
    if reason == "content_filter":
        return "error", "Provider finish_reason: content_filter"
    return "error", f"Provider finish_reason: {reason}"


# ---------------------------------------------------------------------------
# Outgoing conversion (Message[] -> OpenAI chat messages)
# ---------------------------------------------------------------------------
def convert_messages(model: Model, context: Context, compat: OpenAICompat) -> List[Dict[str, Any]]:
    params: List[Dict[str, Any]] = []

    if context.systemPrompt:
        params.append({"role": "system", "content": sanitize_surrogates(context.systemPrompt)})

    for msg in context.messages:
        role = getattr(msg, "role", None)
        if role == "user":
            content = msg.content  # type: ignore[attr-defined]
            if isinstance(content, str):
                if content.strip():
                    params.append({"role": "user", "content": sanitize_surrogates(content)})
            else:
                parts = _to_content_parts(content)
                if parts:
                    params.append({"role": "user", "content": parts})
        elif role == "assistant":
            assistant_msg: Dict[str, Any] = {"role": "assistant"}
            text_parts = [
                b for b in msg.content if getattr(b, "type", None) == "text" and b.text.strip()  # type: ignore[attr-defined]
            ]
            assistant_text = sanitize_surrogates("\n".join(b.text for b in text_parts))
            thinking_blocks = [b for b in msg.content if getattr(b, "type", None) == "thinking"]  # type: ignore[attr-defined]

            if thinking_blocks and not compat.requires_reasoning_content_on_assistant:
                # Replay thinking as plain text (no tags) to avoid the model mimicking them.
                thinking_text = sanitize_surrogates("\n\n".join(b.thinking for b in thinking_blocks))  # type: ignore[attr-defined]
                assistant_msg["content"] = (
                    [{"type": "text", "text": thinking_text}] + [{"type": "text", "text": assistant_text}]
                    if assistant_text
                    else thinking_text
                )
            elif assistant_text:
                assistant_msg["content"] = assistant_text
            else:
                assistant_msg["content"] = None

            if compat.requires_reasoning_content_on_assistant and getattr(model, "reasoning", False):
                assistant_msg["reasoning_content"] = ""

            tool_calls = [b for b in msg.content if getattr(b, "type", None) == "toolCall"]  # type: ignore[attr-defined]
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,  # type: ignore[attr-defined]
                        "type": "function",
                        "function": {
                            "name": tc.name,  # type: ignore[attr-defined]
                            "arguments": json.dumps(tc.arguments or {}, ensure_ascii=False),  # type: ignore[attr-defined]
                        },
                    }
                    for tc in tool_calls
                ]
            if assistant_msg.get("content") is None and "tool_calls" not in assistant_msg:
                # Skip empty assistant messages (some providers reject them).
                continue
            params.append(assistant_msg)
        elif role == "toolResult":
            text_result = "\n".join(
                b.text for b in msg.content if getattr(b, "type", None) == "text"  # type: ignore[attr-defined]
            )
            has_images = any(getattr(b, "type", None) == "image" for b in msg.content)  # type: ignore[attr-defined]
            tool_text = text_result if text_result else (has_images and "(see attached image)") or "(no tool output)"
            params.append(
                {
                    "role": "tool",
                    "content": sanitize_surrogates(tool_text),
                    "tool_call_id": msg.toolCallId,  # type: ignore[attr-defined]
                }
            )
    return params


def _to_content_parts(content: List[Any]) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    for item in content:
        if getattr(item, "type", None) == "text":
            if item.text.strip():
                parts.append({"type": "text", "text": sanitize_surrogates(item.text)})
        else:
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{item.mimeType};base64,{item.data}"},
                }
            )
    return parts


def convert_tools(tools: List[Tool]) -> List[Dict[str, Any]]:
    out = []
    for tool in tools:
        schema = tool.parameters or {}
        out.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": schema.get("properties", {}),
                        "required": schema.get("required", []),
                    },
                },
            }
        )
    return out


def _resolve_base_url(model: Model) -> str:
    if model.baseUrl:
        return model.baseUrl.rstrip("/")
    default = PROVIDER_DEFAULTS.get(model.provider, {}).get("base_url")
    if default:
        return default
    return "https://api.openai.com/v1"


def _resolve_api_key(model: Model, options: Optional[StreamOptions]) -> Optional[str]:
    api_key = getattr(options, "apiKey", None) if options else None
    if api_key:
        return api_key
    headers = getattr(options, "headers", None) if options else None
    if headers:
        auth = headers.get("authorization") or headers.get("Authorization")
        if auth and auth.lower().startswith("bearer "):
            return auth[7:].strip()
    env_var = PROVIDER_DEFAULTS.get(model.provider, {}).get("auth_env")
    if env_var:
        return os.environ.get(env_var)
    return None


def build_params(
    model: Model,
    context: Context,
    options: Optional[StreamOptions],
    compat: OpenAICompat,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "model": model.id,
        "messages": convert_messages(model, context, compat),
        "stream": True,
    }
    if compat.supports_usage_in_streaming:
        params["stream_options"] = {"include_usage": True}

    max_tokens = options.maxTokens if options and options.maxTokens else model.maxTokens  # type: ignore[attr-defined]
    params[compat.max_tokens_field] = max_tokens

    if options and options.temperature is not None:
        params["temperature"] = options.temperature

    if context.tools:
        params["tools"] = convert_tools(context.tools)
        if options and options.toolChoice:
            params["tool_choice"] = options.toolChoice

    reasoning_effort = getattr(options, "reasoningEffort", None) if options else None
    if getattr(model, "reasoning", False):
        _apply_reasoning(params, compat, model, reasoning_effort)

    return params


def _apply_reasoning(params: Dict[str, Any], compat: OpenAICompat, model: Model, effort: Optional[str]) -> None:
    """Apply provider-specific reasoning params (mirrors upstream thinkingFormat switch)."""
    level_map = model.thinkingLevelMap or {}
    mapped = level_map.get(effort or "off") if effort else level_map.get("off")

    if compat.thinking_format == "deepseek":
        if effort:
            params["thinking"] = {"type": "enabled"}
            if mapped and isinstance(mapped, str):
                params["reasoning_effort"] = mapped
        elif level_map.get("off") is not None:
            params["thinking"] = {"type": "disabled"}
    elif compat.thinking_format == "qwen":
        params["enable_thinking"] = bool(effort)
        if effort and mapped and isinstance(mapped, str):
            params["reasoning_effort"] = mapped
    elif compat.thinking_format == "openrouter":
        if effort:
            params["reasoning"] = {"effort": mapped if isinstance(mapped, str) else effort}
        elif level_map.get("off") is not None:
            params["reasoning"] = {"effort": mapped if isinstance(mapped, str) else "none"}
    else:  # openai-style reasoning_effort
        if effort and mapped and isinstance(mapped, str):
            params["reasoning_effort"] = mapped
        elif effort:
            params["reasoning_effort"] = effort
        elif isinstance(level_map.get("off"), str):
            params["reasoning_effort"] = level_map["off"]


# ---------------------------------------------------------------------------
# SSE parsing (decoupled from httpx for testability)
# ---------------------------------------------------------------------------
async def iter_openai_sse(lines: AsyncIterator[str]) -> AsyncIterator[Dict[str, Any]]:
    """Yield parsed JSON chunks from an OpenAI-style SSE line stream.

    ``data: [DONE]`` terminates the stream; ``data: {json}`` lines are parsed.
    """
    async for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            return
        try:
            yield parse_json_with_repair(payload)
        except Exception as e:  # pragma: no cover - malformed line
            raise RuntimeError(f"Could not parse OpenAI SSE data: {e}; data={payload}") from e


# ---------------------------------------------------------------------------
# Chunk mapping (the core streaming event loop)
# ---------------------------------------------------------------------------
async def _map_openai_chunks(
    chunks: AsyncIterator[Dict[str, Any]],
    model: Model,
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
    compat: OpenAICompat,
) -> Tuple[bool, bool]:
    """Consume parsed chunks, push AssistantMessageEvents, return (has_finish_reason, has_tool_use)."""
    blocks = output.content
    text_block: Optional[TextContent] = None
    thinking_block: Optional[ThinkingContent] = None
    tool_blocks_by_index: Dict[int, ToolCall] = {}
    tool_blocks_by_id: Dict[str, ToolCall] = {}
    has_finish_reason = False

    def get_index(block: Any) -> int:
        return blocks.index(block)

    def ensure_text() -> TextContent:
        nonlocal text_block
        if text_block is None:
            text_block = TextContent(text="")
            blocks.append(text_block)
            stream.push(AssistantMessageEvent(type="text_start", contentIndex=get_index(text_block), partial=output))
        return text_block

    def ensure_thinking() -> ThinkingContent:
        nonlocal thinking_block
        if thinking_block is None:
            thinking_block = ThinkingContent(thinking="", thinkingSignature="")
            blocks.append(thinking_block)
            stream.push(
                AssistantMessageEvent(type="thinking_start", contentIndex=get_index(thinking_block), partial=output)
            )
        return thinking_block

    def ensure_tool(delta: Dict[str, Any]) -> ToolCall:
        stream_index = delta.get("index")
        name = (delta.get("function") or {}).get("name") or (delta.get("custom") or {}).get("name") or ""
        block = tool_blocks_by_index.get(stream_index) if isinstance(stream_index, int) else None
        if block is None and delta.get("id"):
            block = tool_blocks_by_id.get(delta["id"])
        if block is None:
            block = ToolCall(id=delta.get("id") or "", name=name, arguments={})
            block.partialArgs = ""  # type: ignore[attr-defined]
            blocks.append(block)
            if isinstance(stream_index, int):
                tool_blocks_by_index[stream_index] = block
            if delta.get("id"):
                tool_blocks_by_id[delta["id"]] = block
            stream.push(AssistantMessageEvent(type="toolcall_start", contentIndex=get_index(block), partial=output))
        if not block.name and name:
            block.name = name
        return block

    def finish_block(block: Any) -> None:
        idx = get_index(block)
        if idx == -1:
            return
        if block.type == "text":
            stream.push(AssistantMessageEvent(type="text_end", contentIndex=idx, content=block.text, partial=output))
        elif block.type == "thinking":
            stream.push(
                AssistantMessageEvent(type="thinking_end", contentIndex=idx, content=block.thinking, partial=output)
            )
        elif block.type == "toolCall":
            block.arguments = parse_streaming_json(getattr(block, "partialArgs", "") or "")  # type: ignore[attr-defined]
            del block.partialArgs  # type: ignore[attr-defined]
            stream.push(AssistantMessageEvent(type="toolcall_end", contentIndex=idx, toolCall=block, partial=output))

    async for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        if chunk.get("id"):
            output.responseId = output.responseId or chunk["id"]
        if chunk.get("model"):
            output.responseModel = output.responseModel or chunk["model"]
        if chunk.get("usage"):
            _parse_chunk_usage(chunk["usage"], model, output)

        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0]
        if not isinstance(choice, dict):
            continue

        if not chunk.get("usage") and isinstance(choice.get("usage"), dict):
            _parse_chunk_usage(choice["usage"], model, output)

        if choice.get("finish_reason"):
            output.rawStopReason = choice["finish_reason"]
            stop_reason, error_msg = map_stop_reason(choice["finish_reason"])
            output.stopReason = stop_reason  # type: ignore[assignment]
            if error_msg:
                output.errorMessage = error_msg
            has_finish_reason = True

        delta = choice.get("delta") or {}
        if isinstance(delta.get("content"), str) and delta["content"]:
            block = ensure_text()
            block.text += delta["content"]
            stream.push(
                AssistantMessageEvent(
                    type="text_delta", contentIndex=get_index(block), delta=delta["content"], partial=output
                )
            )

        for field_name in REASONING_FIELDS:
            value = delta.get(field_name)
            if isinstance(value, str) and value:
                block = ensure_thinking()
                block.thinking += value
                stream.push(
                    AssistantMessageEvent(
                        type="thinking_delta", contentIndex=get_index(block), delta=value, partial=output
                    )
                )
                break

        for tc in delta.get("tool_calls") or []:
            block = ensure_tool(tc)
            if not block.id and tc.get("id"):
                block.id = tc["id"]
            name = (tc.get("function") or {}).get("name")
            if not block.name and name:
                block.name = name
            func_args = (tc.get("function") or {}).get("arguments")
            if isinstance(func_args, str) and func_args:
                block.partialArgs = (getattr(block, "partialArgs", "") or "") + func_args  # type: ignore[attr-defined]
                block.arguments = parse_streaming_json(getattr(block, "partialArgs", ""))  # type: ignore[attr-defined]
                stream.push(
                    AssistantMessageEvent(
                        type="toolcall_delta", contentIndex=get_index(block), delta=func_args, partial=output
                    )
                )

    for block in list(blocks):
        finish_block(block)

    return has_finish_reason, any(getattr(b, "type", None) == "toolCall" for b in blocks)


def _parse_chunk_usage(raw: Dict[str, Any], model: Model, output: AssistantMessage) -> None:
    prompt = raw.get("prompt_tokens") or 0
    cache_read = raw.get("prompt_tokens_details", {}).get("cached_tokens") or raw.get("prompt_cache_hit_tokens") or 0
    cache_write = raw.get("prompt_tokens_details", {}).get("cache_write_tokens") or 0
    inp = max(0, prompt - cache_read - cache_write)
    out = raw.get("completion_tokens") or 0
    output.usage.input = inp
    output.usage.output = out
    output.usage.cacheRead = cache_read
    output.usage.cacheWrite = cache_write
    output.usage.reasoning = raw.get("completion_tokens_details", {}).get("reasoning_tokens") or 0
    output.usage.totalTokens = inp + out + cache_read + cache_write
    calculate_cost(model, output.usage)


# ---------------------------------------------------------------------------
# Stream implementation
# ---------------------------------------------------------------------------
def stream(
    model: Model,
    context: Context,
    options: Optional[StreamOptions] = None,
) -> AssistantMessageEventStream:
    loop = asyncio.get_event_loop()
    stream_obj = AssistantMessageEventStream(loop=loop)
    loop.create_task(_run_stream(model, context, options, stream_obj))
    return stream_obj


async def _run_stream(
    model: Model,
    context: Context,
    options: Optional[StreamOptions],
    stream_obj: AssistantMessageEventStream,
) -> None:
    compat = detect_compat(model)
    output = AssistantMessage(
        role="assistant",
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=_empty_usage(),
        stopReason="pending",  # type: ignore[assignment]
        timestamp=0,
    )

    try:
        api_key = _resolve_api_key(model, options)
        if not api_key:
            raise ValueError(f"No API key for provider: {model.provider}")
        params = build_params(model, context, options, compat)

        request_headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {api_key}",
        }
        headers = getattr(options, "headers", None) if options else None
        if headers:
            for k, v in headers.items():
                if v is not None and k.lower() != "authorization":
                    request_headers[k] = v

        url = _resolve_base_url(model) + "/chat/completions"
        timeout = (options.timeout / 1000.0) if options and options.timeout else 600.0

        stream_obj.push(AssistantMessageEvent(type="start", partial=output))

        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, headers=request_headers, json=params, timeout=timeout) as resp:
                if options and options.onResponse:
                    await _maybe_await(
                        options.onResponse({"status": resp.status_code, "headers": dict(resp.headers)}, model)
                    )
                chunks = iter_openai_sse(resp.aiter_lines())
                has_finish, has_tool = await _map_openai_chunks(chunks, model, output, stream_obj, compat)

        if options and options.signal and getattr(options.signal, "is_set", lambda: False)():
            raise asyncio.CancelledError("Request was aborted")
        if not has_finish and not compat.supports_usage_in_streaming:
            output.stopReason = "toolUse" if has_tool else "stop"  # type: ignore[assignment]
        if output.stopReason == "pending":  # type: ignore[comparison-overlap]
            output.stopReason = "toolUse" if has_tool else "stop"  # type: ignore[assignment]
        if output.stopReason == "error":  # type: ignore[comparison-overlap]
            raise RuntimeError(output.errorMessage or "Provider returned an error stop reason")

        stream_obj.push(AssistantMessageEvent(type="done", reason=output.stopReason, partial=output))
        stream_obj.end()
    except Exception as e:
        for block in output.content:
            if hasattr(block, "partialArgs"):
                del block.partialArgs  # type: ignore[attr-defined]
        aborted = bool(options and options.signal and getattr(options.signal, "is_set", lambda: False)())
        output.stopReason = "aborted" if aborted else "error"  # type: ignore[assignment]
        output.errorMessage = str(e)
        stream_obj.push(AssistantMessageEvent(type="error", reason=output.stopReason, error=output))
        stream_obj.end()


def stream_simple(
    model: Model,
    context: Context,
    options: Optional[SimpleStreamOptions] = None,
) -> AssistantMessageEventStream:
    if options is None or not options.reasoning:
        return stream(model, context, options)
    # Map reasoning level -> reasoning_effort string for the engine.
    effort = options.reasoning if isinstance(options.reasoning, str) else "medium"
    from ..options import StreamOptions as _Full

    full = _Full(
        apiKey=options.apiKey,
        headers=options.headers,
        maxTokens=options.maxTokens,
        temperature=options.temperature,
        toolChoice=options.toolChoice,
        signal=options.signal,
        env=options.env,
        metadata=options.metadata,
        onResponse=options.onResponse,
        onPayload=options.onPayload,
        reasoningEffort=effort,
    )
    return stream(model, context, full)


def _empty_usage():
    from ..types import Usage

    return Usage()


async def _maybe_await(value: Any) -> None:
    if asyncio.iscoroutine(value):
        await value


__all__ = ["stream", "stream_simple", "map_stop_reason", "convert_messages", "convert_tools", "build_params", "detect_compat", "iter_openai_sse"]
