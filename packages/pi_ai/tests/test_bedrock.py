"""Tests for the AWS Bedrock Converse Stream engine (SigV4 + event mapping)."""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any, AsyncIterator, Dict, List

import pytest

from pi_ai.api.bedrock import _map_bedrock_events, aws_sigv4_headers, map_stop_reason
from pi_ai.event_stream import AssistantMessageEventStream
from pi_ai.types import AssistantMessage, Model


def _model() -> Model:
    return Model(id="anthropic.claude-v2", name="claude", api="bedrock-converse-stream", provider="amazon-bedrock", maxTokens=8192)


async def _feed(events, model, out, stream):
    async def gen() -> AsyncIterator[Dict[str, Any]]:
        for e in events:
            yield e

    await _map_bedrock_events(gen(), model, out, stream)


async def _collect(stream):
    stream.end()
    out = []
    async for e in stream:
        out.append(e)
    return out


@pytest.mark.asyncio
async def test_text_and_tool_call_mapping():
    model = _model()
    out = AssistantMessage(role="assistant", content=[], api=model.api, provider=model.provider, model=model.id)
    stream = AssistantMessageEventStream(loop=asyncio.get_event_loop())
    await _feed(
        [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockStart": {"contentBlockIndex": 0, "start": {"text": ""}}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Hello"}}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"contentBlockStart": {"contentBlockIndex": 1, "start": {"toolUse": {"toolUseId": "t1", "name": "grep"}}}},
            {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"toolUse": {"input": '{"pattern":"TODO"'}}}},
            {"contentBlockStop": {"contentBlockIndex": 1}},
            {"messageStop": {"stopReason": "tool_use"}},
            {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5, "cacheReadInputTokens": 2, "totalTokens": 17}}},
        ],
        model,
        out,
        stream,
    )
    assert out.stopReason == "toolUse"
    texts = [b for b in out.content if getattr(b, "type", None) == "text"]
    tcs = [b for b in out.content if getattr(b, "type", None) == "toolCall"]
    assert texts and texts[0].text == "Hello"
    assert tcs and tcs[0].name == "grep" and tcs[0].arguments == {"pattern": "TODO"}
    # usage.input is net of cache reads, matching the shared OpenAI-compat
    # convention (input = inputTokens - cacheReadInputTokens).
    assert out.usage.input == 8
    assert out.usage.cacheRead == 2


def test_sigv4_headers_structure():
    url = "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-v2/converse-stream"
    body = json.dumps({"messages": []}).encode("utf-8")
    headers = aws_sigv4_headers("POST", url, body, "AKIDEXAMPLE", "SECRETKEY", "us-east-1", "bedrock")
    assert "authorization" in headers
    assert headers["authorization"].startswith("AWS4-HMAC-SHA256")
    assert "Credential=AKIDEXAMPLE/" in headers["authorization"]
    assert "Signature=" in headers["authorization"]
    assert "x-amz-date" in headers
    assert headers["x-amz-content-sha256"] == hashlib.sha256(body).hexdigest()
    # host is signed, not the raw authorization value
    assert headers["host"] == "bedrock-runtime.us-east-1.amazonaws.com"


def test_sigv4_with_session_token():
    url = "https://bedrock-runtime.us-west-2.amazonaws.com/model/x/converse-stream"
    body = b"{}"
    headers = aws_sigv4_headers("POST", url, body, "AKID", "SECRET", "us-west-2", "bedrock", token="TOKEN123")
    assert headers["x-amz-security-token"] == "TOKEN123"
    # The token is a signed header (in SignedHeaders), not embedded in the
    # Credential string — assert it is preserved and covered by the signature.
    assert "x-amz-security-token" in headers["authorization"]


def test_map_stop_reason():
    assert map_stop_reason("end_turn") == ("stop", None)
    assert map_stop_reason("max_tokens") == ("length", None)
    assert map_stop_reason("tool_use") == ("toolUse", None)
    sr, _ = map_stop_reason("content_filtered")
    assert sr == "error"
