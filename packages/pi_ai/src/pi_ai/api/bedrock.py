"""AWS Bedrock Converse Stream adapter (Python port of packages/ai/src/api/bedrock-converse-stream.ts).

Streams from the Bedrock Runtime Converse Stream HTTP endpoint and maps its
event stream (newline-delimited JSON objects) into the same
``AssistantMessageEvent`` sequence. Requests are signed with AWS Signature V4
(pure-Python signer below — no ``boto3``/``aioboto3`` dependency, keeping the
port self-contained and consistent with the httpx-based siblings).

Pragmatic simplifications vs upstream: profile/env credential chains, proxy
agents, deferred tools, data-retention diagnostics, and the bearer-token path
are omitted. Standard access-key/secret (optionally session token) auth and
SigV4 signing are implemented.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

import httpx

from ..event_stream import AssistantMessageEventStream
from ..models import calculate_cost
from ..options import BedrockOptions, SimpleStreamOptions, StreamOptions
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
from ..utils.json_parse import parse_streaming_json
from ..utils.text import sanitize_surrogates

AWS_REGIONS_FALLBACK = "us-east-1"


def map_stop_reason(reason: Optional[str]) -> Tuple[str, Optional[str]]:
    """Maps a Bedrock Converse stopReason to pi's StopReason."""
    if reason is None:
        return "pending", None  # type: ignore[return-value]
    if reason == "end_turn":
        return "stop", None
    if reason == "max_tokens":
        return "length", None
    if reason == "tool_use":
        return "toolUse", None
    if reason == "content_filtered":
        return "error", "Provider stopped with: content_filtered"
    return "error", f"Provider stop reason: {reason}"


# ---------------------------------------------------------------------------
# AWS SigV4 signing (compact, dependency-free)
# ---------------------------------------------------------------------------
def _hmac(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _hash_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def aws_sigv4_headers(
    method: str,
    url: str,
    body: bytes,
    access_key: str,
    secret_key: str,
    region: str,
    service: str = "bedrock",
    token: Optional[str] = None,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Return the full header dict (including Authorization) for a SigV4 request."""
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    path = parsed.path or "/"
    query = parsed.query

    now = datetime.now(timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")

    payload_hash = _hash_hex(body)
    headers_to_sign: Dict[str, str] = {"host": host, "x-amz-content-sha256": payload_hash, "x-amz-date": amz_date}
    if token:
        headers_to_sign["x-amz-security-token"] = token
    if extra_headers:
        for k, v in extra_headers.items():
            if k.lower() not in ("authorization", "host"):
                headers_to_sign[k] = v

    sorted_headers = dict(sorted(headers_to_sign.items()))
    canonical_headers = "".join(f"{k}:{sorted_headers[k]}\n" for k in sorted_headers)
    signed_headers = ";".join(sorted_headers.keys())

    canonical_request = "\n".join([method, path, query, canonical_headers, signed_headers, payload_hash])

    credential_scope = f"{date_stamp}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join(
        ["AWS4-HMAC-SHA256", amz_date, credential_scope, _hash_hex(canonical_request.encode("utf-8"))]
    )

    k_date = _hmac(("AWS4" + secret_key).encode("utf-8"), date_stamp)
    k_region = _hmac(k_date, region)
    k_service = _hmac(k_region, service)
    k_signing = _hmac(k_service, "aws4_request")
    signature = hmac.new(k_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    authorization = (
        f"AWS4-HMAC-SHA256 Credential={access_key}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )
    out = dict(sorted_headers)
    out["authorization"] = authorization
    return out


# ---------------------------------------------------------------------------
# Outgoing conversion (Message[] -> Bedrock Converse format)
# ---------------------------------------------------------------------------
def convert_messages(context: Context) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    for msg in context.messages:
        role = getattr(msg, "role", None)
        if role == "user":
            content = msg.content  # type: ignore[attr-defined]
            if isinstance(content, str):
                if content.strip():
                    messages.append({"role": "user", "content": [{"text": sanitize_surrogates(content)}]})
            else:
                parts = _user_parts(content)
                if parts:
                    messages.append({"role": "user", "content": parts})
        elif role == "assistant":
            parts: List[Dict[str, Any]] = []
            for block in msg.content:  # type: ignore[attr-defined]
                btype = getattr(block, "type", None)
                if btype == "text" and block.text.strip():  # type: ignore[attr-defined]
                    parts.append({"text": sanitize_surrogates(block.text)})  # type: ignore[attr-defined]
                elif btype == "thinking" and block.thinking.strip():  # type: ignore[attr-defined]
                    parts.append({"text": sanitize_surrogates(block.thinking)})  # type: ignore[attr-defined]
                elif btype == "toolCall":
                    parts.append(
                        {
                            "toolUse": {
                                "toolUseId": block.id,  # type: ignore[attr-defined]
                                "name": block.name,  # type: ignore[attr-defined]
                                "input": block.arguments or {},  # type: ignore[attr-defined]
                            }
                        }
                    )
            if parts:
                messages.append({"role": "assistant", "content": parts})
        elif role == "toolResult":
            text_result = "\n".join(
                b.text for b in msg.content if getattr(b, "type", None) == "text"  # type: ignore[attr-defined]
            )
            has_images = any(getattr(b, "type", None) == "image" for b in msg.content)  # type: ignore[attr-defined]
            tool_text = text_result if text_result else (has_images and "(see attached image)") or ""
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "toolResult": {
                                "toolUseId": msg.toolCallId,  # type: ignore[attr-defined]
                                "content": [{"text": sanitize_surrogates(tool_text)}],
                                "status": "error" if msg.isError else "success",  # type: ignore[attr-defined]
                            }
                        }
                    ],
                }
            )
    return messages


def _user_parts(content: List[Any]) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    for item in content:
        if getattr(item, "type", None) == "text":
            if item.text.strip():
                parts.append({"text": sanitize_surrogates(item.text)})
        else:
            parts.append({"image": {"format": _fmt(item.mimeType), "source": {"bytes": item.data}}})
    return parts


def _fmt(mime: str) -> str:
    return (mime or "image/png").split("/")[-1]


def convert_tool_config(tools: List[Tool], tool_choice: Any) -> Dict[str, Any]:
    schema_configs = []
    for tool in tools:
        schema = tool.parameters or {}
        schema_configs.append(
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": schema.get("properties", {}),
                        "required": schema.get("required", []),
                    }
                },
            }
        )
    config: Dict[str, Any] = {"tools": schema_configs}
    if isinstance(tool_choice, str):
        config["toolChoice"] = {"auto": {}, "any": {"any": {}}, "none": {"any": {}}}.get(tool_choice, {"auto": {}})
    elif tool_choice is not None:
        config["toolChoice"] = tool_choice
    return config


def build_params(model: Model, context: Context, options: Optional[BedrockOptions]) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "inferenceConfig": {},
        "messages": convert_messages(context),
    }
    if context.systemPrompt:
        params["system"] = [{"text": sanitize_surrogates(context.systemPrompt)}]
    max_tokens = (options.maxTokens if options and options.maxTokens else model.maxTokens)
    params["inferenceConfig"]["maxTokens"] = max_tokens
    if options and options.temperature is not None:
        params["inferenceConfig"]["temperature"] = options.temperature
    if context.tools:
        params["toolConfig"] = convert_tool_config(context.tools, options.toolChoice if options else None)
    return params


def _resolve_region(model: Model) -> str:
    arn = model.id
    m = arn.match(r"^arn:aws(?:-[a-z0-9-]+)?:bedrock:([a-z0-9-]+):")
    if m:
        return m.group(1)
    if model.baseUrl:
        host = urlsplit(model.baseUrl).hostname or ""
        parts = host.split(".")
        for i, p in enumerate(parts):
            if p == "bedrock-runtime" and i + 1 < len(parts):
                return parts[i + 1]
    return AWS_REGIONS_FALLBACK


def _resolve_endpoint(model: Model, region: str) -> str:
    if model.baseUrl:
        return model.baseUrl.rstrip("/")
    return f"https://bedrock-runtime.{region}.amazonaws.com"


# ---------------------------------------------------------------------------
# Event stream parsing (newline-delimited JSON objects)
# ---------------------------------------------------------------------------
async def iter_bedrock_events(lines: AsyncIterator[str]) -> AsyncIterator[Dict[str, Any]]:
    async for raw in lines:
        line = raw.strip()
        if not line:
            continue
        try:
            yield parse_json(line)
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"Could not parse Bedrock event: {e}; line={line}") from e


def parse_json(line: str) -> Dict[str, Any]:
    return json.loads(line)


# ---------------------------------------------------------------------------
# Event mapping
# ---------------------------------------------------------------------------
async def _map_bedrock_events(
    events: AsyncIterator[Dict[str, Any]],
    model: Model,
    output: AssistantMessage,
    stream_obj: AssistantMessageEventStream,
) -> None:
    blocks = output.content

    def get_index(block: Any) -> int:
        return blocks.index(block)

    async for item in events:
        if not isinstance(item, dict):
            continue
        if item.get("messageStart"):
            stream_obj.push(AssistantMessageEvent(type="start", partial=output))
        elif item.get("contentBlockStart"):
            start = item["contentBlockStart"].get("start") or {}
            if start.get("toolUse"):
                block = ToolCall(
                    id=start["toolUse"].get("toolUseId") or "",
                    name=start["toolUse"].get("name") or "",
                    arguments={},
                )
                block.partialJson = ""  # type: ignore[attr-defined]
                block.index = item["contentBlockStart"].get("contentBlockIndex")  # type: ignore[attr-defined]
                blocks.append(block)
                stream_obj.push(
                    AssistantMessageEvent(type="toolcall_start", contentIndex=get_index(block), partial=output)
                )
        elif item.get("contentBlockDelta"):
            delta = item["contentBlockDelta"].get("delta") or {}
            idx = item["contentBlockDelta"].get("contentBlockIndex")
            block = _find_block(blocks, idx)
            if isinstance(delta.get("text"), str):
                if block is None or getattr(block, "type", None) != "text":
                    new_block = TextContent(text="")
                    new_block.index = idx  # type: ignore[attr-defined]
                    blocks.append(new_block)
                    block = new_block
                    stream_obj.push(AssistantMessageEvent(type="text_start", contentIndex=get_index(block), partial=output))
                block.text += delta["text"]  # type: ignore[attr-defined]
                stream_obj.push(
                    AssistantMessageEvent(
                        type="text_delta", contentIndex=get_index(block), delta=delta["text"], partial=output
                    )
                )
            elif isinstance(delta.get("toolUse"), dict):
                if block is not None and getattr(block, "type", None) == "toolCall":
                    inp = delta["toolUse"].get("input") or ""
                    block.partialJson = (getattr(block, "partialJson", "") or "") + inp  # type: ignore[attr-defined]
                    block.arguments = parse_streaming_json(getattr(block, "partialJson", ""))  # type: ignore[attr-defined]
                    stream_obj.push(
                        AssistantMessageEvent(
                            type="toolcall_delta", contentIndex=get_index(block), delta=inp, partial=output
                        )
                    )
            elif delta.get("reasoningContent") and isinstance(delta["reasoningContent"].get("text"), str):
                if block is None or getattr(block, "type", None) != "thinking":
                    new_block = ThinkingContent(thinking="", thinkingSignature=None)
                    new_block.index = idx  # type: ignore[attr-defined]
                    blocks.append(new_block)
                    block = new_block
                    stream_obj.push(
                        AssistantMessageEvent(type="thinking_start", contentIndex=get_index(block), partial=output)
                    )
                block.thinking += delta["reasoningContent"]["text"]  # type: ignore[attr-defined]
                stream_obj.push(
                    AssistantMessageEvent(
                        type="thinking_delta",
                        contentIndex=get_index(block),
                        delta=delta["reasoningContent"]["text"],
                        partial=output,
                    )
                )
        elif item.get("contentBlockStop"):
            idx = item["contentBlockStop"].get("contentBlockIndex")
            block = _find_block(blocks, idx)
            if block is not None:
                if hasattr(block, "index"):
                    del block.index  # type: ignore[attr-defined]
                btype = getattr(block, "type", None)
                if btype == "text":
                    stream_obj.push(
                        AssistantMessageEvent(type="text_end", contentIndex=get_index(block), content=block.text, partial=output)
                    )
                elif btype == "thinking":
                    stream_obj.push(
                        AssistantMessageEvent(
                            type="thinking_end", contentIndex=get_index(block), content=block.thinking, partial=output
                        )
                    )
                elif btype == "toolCall":
                    block.arguments = parse_streaming_json(getattr(block, "partialJson", "") or "")  # type: ignore[attr-defined]
                    if hasattr(block, "partialJson"):
                        del block.partialJson  # type: ignore[attr-defined]
                    stream_obj.push(
                        AssistantMessageEvent(
                            type="toolcall_end", contentIndex=get_index(block), toolCall=block, partial=output
                        )
                    )
        elif item.get("messageStop"):
            stop = item["messageStop"].get("stopReason")
            output.rawStopReason = stop
            reason, error_msg = map_stop_reason(stop)
            output.stopReason = reason  # type: ignore[assignment]
            if error_msg:
                output.errorMessage = error_msg
        elif item.get("metadata"):
            usage = item["metadata"].get("usage") or {}
            inp = (usage.get("inputTokens") or 0) - (usage.get("cacheReadInputTokens") or 0)
            out = usage.get("outputTokens") or 0
            output.usage.input = inp
            output.usage.output = out
            output.usage.cacheRead = usage.get("cacheReadInputTokens") or 0
            output.usage.totalTokens = usage.get("totalTokens") or 0
            calculate_cost(model, output.usage)


def _find_block(blocks: List[Any], index: Any) -> Any:
    if index is None:
        return None
    for b in blocks:
        if getattr(b, "index", None) == index:
            return b
    return None


# ---------------------------------------------------------------------------
# Stream implementation
# ---------------------------------------------------------------------------
def stream(
    model: Model,
    context: Context,
    options: Optional[BedrockOptions] = None,
) -> AssistantMessageEventStream:
    loop = asyncio.get_event_loop()
    stream_obj = AssistantMessageEventStream(loop=loop)
    loop.create_task(_run_stream(model, context, options, stream_obj))
    return stream_obj


async def _run_stream(
    model: Model,
    context: Context,
    options: Optional[BedrockOptions],
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
        access_key = _env("AWS_ACCESS_KEY_ID", options)
        secret_key = _env("AWS_SECRET_ACCESS_KEY", options)
        token = _env("AWS_SESSION_TOKEN", options)
        if not (access_key and secret_key):
            raise ValueError("No AWS credentials for Bedrock (set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)")

        region = _resolve_region(model)
        endpoint = _resolve_endpoint(model, region)
        url = f"{endpoint}/model/{model.id}/converse-stream"
        params = build_params(model, context, options)
        body = json.dumps(params).encode("utf-8")

        request_headers = aws_sigv4_headers(
            "POST", url, body, access_key, secret_key, region, "bedrock", token=token
        )
        request_headers["content-type"] = "application/json"

        timeout = (options.timeout / 1000.0) if options and options.timeout else 600.0

        async with httpx.AsyncClient() as client:
            async with client.stream("POST", url, headers=request_headers, content=body, timeout=timeout) as resp:
                if options and options.onResponse:
                    await _maybe_await(
                        options.onResponse({"status": resp.status_code, "headers": dict(resp.headers)}, model)
                    )
                events = iter_bedrock_events(resp.aiter_lines())
                await _map_bedrock_events(events, model, output, stream_obj)

        if options and options.signal and getattr(options.signal, "is_set", lambda: False)():
            raise asyncio.CancelledError("Request was aborted")
        if output.stopReason == "pending":  # type: ignore[comparison-overlap]
            raise RuntimeError("Bedrock stream ended without a stop reason")
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
    base = BedrockOptions(
        apiKey=options.apiKey if options else None,
        headers=options.headers if options else None,
        maxTokens=options.maxTokens if options else None,
        temperature=options.temperature if options else None,
        toolChoice=options.toolChoice if options else None,
        signal=options.signal if options else None,
        env=options.env if options else None,
        metadata=options.metadata if options else None,
        onResponse=options.onResponse if options else None,
        onPayload=options.onPayload if options else None,
    )
    return stream(model, context, base)


def _env(name: str, options: Optional[StreamOptions]) -> Optional[str]:
    import os

    if options and options.env and options.env.get(name):
        return options.env[name]
    return os.environ.get(name)


def _empty_usage():
    from ..types import Usage

    return Usage()


async def _maybe_await(value: Any) -> None:
    if asyncio.iscoroutine(value):
        await value


__all__ = ["stream", "stream_simple", "map_stop_reason", "convert_messages", "convert_tool_config", "build_params", "aws_sigv4_headers"]
