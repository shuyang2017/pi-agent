"""Tests for the Google (Gemini) streaming engine."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, List

import pytest

from pi_ai.api.google import _map_google_chunks, map_stop_reason
from pi_ai.event_stream import AssistantMessageEventStream
from pi_ai.types import AssistantMessage, Model


def _model() -> Model:
    return Model(id="gemini-1.5-pro", name="gemini", api="google-generative-ai", provider="google", maxTokens=8192)


async def _feed(chunks, model, out, stream):
    async def gen() -> AsyncIterator[Dict[str, Any]]:
        for c in chunks:
            yield c

    await _map_google_chunks(gen(), model, out, stream)


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
            {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "Searching"}]},
                        "finishReason": "STOP",
                    }
                ]
            },
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"functionCall": {"name": "grep", "args": {"pattern": "TODO"}}}
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ]
            },
        ],
        model,
        out,
        stream,
    )
    # Second chunk re-emits candidate with tool call; stopReason should be toolUse.
    assert out.stopReason == "toolUse"
    texts = [b for b in out.content if getattr(b, "type", None) == "text"]
    tcs = [b for b in out.content if getattr(b, "type", None) == "toolCall"]
    assert texts and "Searching" in texts[0].text
    assert tcs and tcs[0].name == "grep" and tcs[0].arguments == {"pattern": "TODO"}


def test_map_stop_reason():
    assert map_stop_reason("STOP") == ("stop", None)
    assert map_stop_reason("MAX_TOKENS") == ("length", None)
    sr, msg = map_stop_reason("SAFETY")
    assert sr == "error" and msg is not None
    assert map_stop_reason(None)[0] == "pending"
