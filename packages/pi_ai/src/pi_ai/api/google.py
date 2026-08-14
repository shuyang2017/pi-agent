"""Google Generative AI (Gemini) adapter (Python port of packages/ai/src/api/google-generative-ai.ts + google-shared.ts).

Streams from the Generative Language REST endpoint
``POST https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse``
and maps Gemini ``Content``/``Part`` streams into the same ``AssistantMessageEvent``
sequence. Auth uses the ``x-goog-api-key`` header (the upstream also supports
OAuth; that path is intentionally simplified, noted in code).

Pragmatic simplifications vs upstream: thought-signature replay across turns,
Gemini-3/Gemma-4 specific thinking-level heuristics, and Cloud Code Assist
merging are omitted; the core text/thinking/tool-call mapping and stop-reason
translation are faithful.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx

from ..event_stream import AssistantMessageEventStream
from ..models import calculate_cost
from ..options import GoogleOptions, GoogleThinkingConfig, SimpleStreamOptions, StreamOptions
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
from ..utils.json_parse import parse_json_with_repair
from ..utils.text import sanitize_surrogates


def map_stop_reason(reason: Optional[str]) -> Tuple[str, Optional[str]]:
    """Maps a Gemini FinishReason to pi's StopReason (mirrors google-shared.mapStopReason)."""
    if reason is None:
        return "pending", None  # type: ignore[return-value]
    if reason == "STOP":
        return "stop", None
    if reason == "MAX_TOKENS":
        return "length", None
    if reason in (
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "SAFETY",
        "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT",
        "IMAGE_RECITATION",
        "IMAGE_OTHER",
        "RECITATION",
        "OTHER",
        "LANGUAGE",
        "MALFORMED_FUNCTION_CALL",
        "UNEXPECTED_TOOL_CALL",
        "NO_IMAGE",
        "FINISH_REASON_UNSPECIFIED",
    ):
        return "error", f"Provider stopped with: {reason}"
    return "error", f"Unhandled stop reason: {reason}"


# ---------------------------------------------------------------------------
# Outgoing conversion (Message[] -> Gemini Content[])
# ---------------------------------------------------------------------------
def convert_messages(model: Model, context: Context) -> List[Dict[str, Any]]:
    contents: List[Dict[str, Any]] = []

    for msg in context.messages:
        role = getattr(msg, "role", None)
        if role == "user":
            content = msg.content  # type: ignore[attr-defined]
            if isinstance(content, str):
                if content.strip():
                    contents.append({"role": "user", "parts": [{"text": sanitize_surrogates(content)}]})
            else:
                parts = _user_parts(content)
                if parts:
                    contents.append({"role": "user", "parts": parts})
        elif role == "assistant":
            parts: List[Dict[str, Any]] = []
            for block in msg.content:  # type: ignore[attr-defined]
                btype = getattr(block, "type", None)
                if btype == "text":
                    if block.text.strip():  # type: ignore[attr-defined]
                        parts.append({"text": sanitize_surrogates(block.text)})  # type: ignore[attr-defined]
                elif btype == "thinking":
                    if block.thinking.strip():  # type: ignore[attr-defined]
                        parts.append({"text": sanitize_surrogates(block.thinking)})  # type: ignore[attr-defined]
                elif btype == "toolCall":
                    parts.append(
                        {
                            "functionCall": {
                                "name": block.name,  # type: ignore[attr-defined]
                                "args": block.arguments or {},  # type: ignore[attr-defined]
                            }
                        }
                    )
            if parts:
                contents.append({"role": "model", "parts": parts})
        elif role == "toolResult":
            text_result = "\n".join(
                b.text for b in msg.content if getattr(b, "type", None) == "text"  # type: ignore[attr-defined]
            )
            has_images = any(getattr(b, "type", None) == "image" for b in msg.content)  # type: ignore[attr-defined]
            response_value = text_result if text_result else (has_images and "(see attached image)") or ""
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": msg.toolName,  # type: ignore[attr-defined]
                                "response": ({"error": response_value} if msg.isError else {"output": response_value}),  # type: ignore[attr-defined]
                            }
                        }
                    ],
                }
            )
    return contents


def _user_parts(content: List[Any]) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    for item in content:
        if getattr(item, "type", None) == "text":
            if item.text.strip():
                parts.append({"text": sanitize_surrogates(item.text)})
        else:
            parts.append({"inlineData": {"mimeType": item.mimeType, "data": item.data}})
    return parts


def convert_tools(tools: List[Tool]) -> List[Dict[str, Any]]:
    if not tools:
        return []
    declarations = []
    for tool in tools:
        schema = tool.parameters or {}
        declarations.append(
            {
                "name": tool.name,
                "description": tool.description,
                "parametersJsonSchema": {
                    "type": "object",
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", []),
                },
            }
        )
    return [{"functionDeclarations": declarations}]


def build_params(model: Model, context: Context, options: Optional[GoogleOptions]) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    if options and options.temperature is not None:
        config["temperature"] = options.temperature
    if options and options.maxTokens is not None:
        config["maxOutputTokens"] = options.maxTokens

    if context.systemPrompt:
        config["systemInstruction"] = {"parts": [{"text": sanitize_surrogates(context.systemPrompt)}]}
    if context.tools:
        config["tools"] = convert_tools(context.tools)

    opts = options or GoogleOptions()
    if getattr(model, "reasoning", False) and getattr(opts, "thinking", None) and opts.thinking.enabled:  # type: ignore[attr-defined]
        thinking_config: Dict[str, Any] = {"includeThoughts": True}
        if opts.thinking.level is not None:  # type: ignore[attr-defined]
            thinking_config["thinkingLevel"] = opts.thinking.level  # type: ignore[attr-defined]
        elif opts.thinking.budgetTokens is not None:  # type: ignore[attr-defined]
            thinking_config["thinkingBudget"] = opts.thinking.budgetTokens  # type: ignore[attr-defined]
        config["thinkingConfig"] = thinking_config

    return {
        "model": f"models/{model.id}",
        "contents": convert_messages(model, context),
        **({"generationConfig": config} if config else {}),
    }


# ---------------------------------------------------------------------------
# SSE parsing + chunk mapping
# ---------------------------------------------------------------------------
async def iter_google_sse(lines: AsyncIterator[str]) -> AsyncIterator[Dict[str, Any]]:
    async for raw in lines:
        line = raw.strip()
        if not line or not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if not payload:
            continue
        try:
            yield parse_json_with_repair(payload)
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"Could not parse Google SSE data: {e}; data={payload}") from e


async def _map_google_chunks(
    chunks: AsyncIterator[Dict[str, Any]],
    model: Model,
    output: AssistantMessage,
    stream_obj: AssistantMessageEventStream,
) -> None:
    blocks = output.content
    current_block: Any = None

    def block_index() -> int:
        return len(blocks) - 1

    def finish_current() -> None:
        nonlocal current_block
        if current_block is None:
            return
        idx = block_index()
        if current_block.type == "text":
            stream_obj.push(
                AssistantMessageEvent(type="text_end", contentIndex=idx, content=current_block.text, partial=output)
            )
        elif current_block.type == "thinking":
            stream_obj.push(
                AssistantMessageEvent(
                    type="thinking_end", contentIndex=idx, content=current_block.thinking, partial=output
                )
            )
        current_block = None

    async for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        output.responseId = output.responseId or chunk.get("responseId")

        candidate = (chunk.get("candidates") or [{}])[0] if chunk.get("candidates") else None
        if candidate and isinstance(candidate, dict) and candidate.get("content", {}).get("parts"):
            for part in candidate["content"]["parts"]:
                if not isinstance(part, dict):
                    continue
                if isinstance(part.get("text"), str):
                    is_thinking = bool(part.get("thought"))
                    if (
                        current_block is None
                        or (is_thinking and current_block.type != "thinking")
                        or (not is_thinking and current_block.type != "text")
                    ):
                        finish_current()
                        if is_thinking:
                            current_block = ThinkingContent(thinking="", thinkingSignature=None)
                            blocks.append(current_block)
                            stream_obj.push(
                                AssistantMessageEvent(
                                    type="thinking_start", contentIndex=block_index(), partial=output
                                )
                            )
                        else:
                            current_block = TextContent(text="")
                            blocks.append(current_block)
                            stream_obj.push(
                                AssistantMessageEvent(type="text_start", contentIndex=block_index(), partial=output)
                            )
                    if current_block.type == "thinking":
                        current_block.thinking += part["text"]
                        stream_obj.push(
                            AssistantMessageEvent(
                                type="thinking_delta",
                                contentIndex=block_index(),
                                delta=part["text"],
                                partial=output,
                            )
                        )
                    else:
                        current_block.text += part["text"]
                        stream_obj.push(
                            AssistantMessageEvent(
                                type="text_delta", contentIndex=block_index(), delta=part["text"], partial=output
                            )
                        )
                elif isinstance(part.get("functionCall"), dict):
                    finish_current()
                    fc = part["functionCall"]
                    tool_call = ToolCall(
                        id=fc.get("id") or f"{fc.get('name', 'tool')}_{len(blocks)}",
                        name=fc.get("name", ""),
                        arguments=fc.get("args") or {},
                    )
                    blocks.append(tool_call)
                    stream_obj.push(
                        AssistantMessageEvent(type="toolcall_start", contentIndex=block_index(), partial=output)
                    )
                    stream_obj.push(
                        AssistantMessageEvent(
                            type="toolcall_delta",
                            contentIndex=block_index(),
                            delta=__import__("json").dumps(tool_call.arguments, ensure_ascii=False),
                            partial=output,
                        )
                    )
                    stream_obj.push(
                        AssistantMessageEvent(
                            type="toolcall_end", contentIndex=block_index(), toolCall=tool_call, partial=output
                        )
                    )

        if candidate and isinstance(candidate, dict) and candidate.get("finishReason"):
            output.rawStopReason = candidate["finishReason"]
            stop_reason, error_msg = map_stop_reason(candidate["finishReason"])
            output.stopReason = stop_reason  # type: ignore[assignment]
            if error_msg:
                output.errorMessage = error_msg
            if any(getattr(b, "type", None) == "toolCall" for b in blocks):
                output.stopReason = "toolUse"  # type: ignore[assignment]

        usage = chunk.get("usageMetadata")
        if isinstance(usage, dict):
            inp = (usage.get("promptTokenCount") or 0) - (usage.get("cachedContentTokenCount") or 0)
            out = (usage.get("candidatesTokenCount") or 0) + (usage.get("thoughtsTokenCount") or 0)
            output.usage.input = inp
            output.usage.output = out
            output.usage.cacheRead = usage.get("cachedContentTokenCount") or 0
            output.usage.reasoning = usage.get("thoughtsTokenCount") or 0
            output.usage.totalTokens = usage.get("totalTokenCount") or 0
            calculate_cost(model, output.usage)

    finish_current()


# ---------------------------------------------------------------------------
# Stream implementation
# ---------------------------------------------------------------------------
def stream(
    model: Model,
    context: Context,
    options: Optional[GoogleOptions] = None,
) -> AssistantMessageEventStream:
    loop = asyncio.get_event_loop()
    stream_obj = AssistantMessageEventStream(loop=loop)
    loop.create_task(_run_stream(model, context, options, stream_obj))
    return stream_obj


async def _run_stream(
    model: Model,
    context: Context,
    options: Optional[GoogleOptions],
    stream_obj: AssistantMessageEventStream,
) -> None:
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
        api_key = options.apiKey if options else None
        if not api_key:
            raise ValueError(f"No API key for provider: {model.provider}")
        params = build_params(model, context, options)

        base = model.baseUrl.rstrip("/") if model.baseUrl else "https://generativelanguage.googleapis.com/v1beta"
        url = f"{base}/models/{model.id}:streamGenerateContent?alt=sse"
        request_headers = {"content-type": "application/json", "x-goog-api-key": api_key}
        headers = getattr(options, "headers", None) if options else None
        if headers:
            for k, v in headers.items():
                if v is not None:
                    request_headers[k] = v
        timeout = (options.timeout / 1000.0) if options and options.timeout else 600.0

        stream_obj.push(AssistantMessageEvent(type="start", partial=output))

        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, headers=request_headers, json=params, timeout=timeout) as resp:
                if options and options.onResponse:
                    await _maybe_await(
                        options.onResponse({"status": resp.status_code, "headers": dict(resp.headers)}, model)
                    )
                chunks = iter_google_sse(resp.aiter_lines())
                await _map_google_chunks(chunks, model, output, stream_obj)

        if options and options.signal and getattr(options.signal, "is_set", lambda: False)():
            raise asyncio.CancelledError("Request was aborted")
        if output.stopReason == "pending":  # type: ignore[comparison-overlap]
            raise RuntimeError("Google stream ended without a finish reason")
        if output.stopReason in ("aborted", "error"):  # type: ignore[comparison-overlap]
            raise RuntimeError(output.errorMessage or "An unknown error occurred")

        stream_obj.push(AssistantMessageEvent(type="done", reason=output.stopReason, partial=output))
        stream_obj.end()
    except Exception as e:
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
        return stream(
            model,
            context,
            GoogleOptions(apiKey=options.apiKey if options else None, thinking=GoogleThinkingConfig(enabled=False))
            if options is None
            else GoogleOptions(
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
                thinking=GoogleThinkingConfig(enabled=False),
            ),
        )
    return stream(
        model,
        context,
        GoogleOptions(
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
            thinking=GoogleThinkingConfig(enabled=True),
        ),
    )


def _empty_usage():
    from ..types import Usage

    return Usage()


async def _maybe_await(value: Any) -> None:
    if asyncio.iscoroutine(value):
        await value


__all__ = ["stream", "stream_simple", "map_stop_reason", "convert_messages", "convert_tools", "build_params"]
