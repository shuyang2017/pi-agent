"""Tests for the OpenAI-compatible streaming engine (OpenRouter/Qwen/DeepSeek share it)."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, List

import pytest

from pi_ai.api._openai_compat import (
    _map_openai_chunks,
    detect_compat,
    map_stop_reason,
)
from pi_ai.event_stream import AssistantMessageEventStream
from pi_ai.types import AssistantMessage, Model


def _model(provider: str, api: str = "openai-completions") -> Model:
    return Model(id="test-model", name="test", api=api, provider=provider, maxTokens=8192)


async def _feed(
    chunks: List[Dict[str, Any]],
    model: Model,
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
):
    async def gen() -> AsyncIterator[Dict[str, Any]]:
        for c in chunks:
            yield c

    compat = detect_compat(model)
    await _map_openai_chunks(gen(), model, output, stream, compat)


async def _collect(stream: AssistantMessageEventStream) -> List[Any]:
    stream.end()
    events = []
    async for e in stream:
        events.append(e)
    return events


def _types(events: List[Any]) -> List[str]:
    return [e.type for e in events]


@pytest.mark.asyncio
async def test_text_mapping_and_stop():
    model = _model("openai")
    out = AssistantMessage(role="assistant", content=[], api=model.api, provider=model.provider, model=model.id)
    stream = AssistantMessageEventStream(loop=asyncio.get_event_loop())
    await _feed(
        [
            {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": " world"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ],
        model,
        out,
        stream,
    )
    assert out.stopReason == "stop"
    texts = [b for b in out.content if getattr(b, "type", None) == "text"]
    assert texts and texts[0].text == "Hello world"
    assert _types(await _collect(stream)) == ["text_start", "text_delta", "text_delta", "text_end"]


@pytest.mark.asyncio
async def test_tool_call_mapping_and_tooluse():
    model = _model("openai")
    out = AssistantMessage(role="assistant", content=[], api=model.api, provider=model.provider, model=model.id)
    stream = AssistantMessageEventStream(loop=asyncio.get_event_loop())
    await _feed(
        [
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "grep", "arguments": ""}}]}, "finish_reason": None}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"pattern":"TODO"'}}]}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ],
        model,
        out,
        stream,
    )
    assert out.stopReason == "toolUse"
    tcs = [b for b in out.content if getattr(b, "type", None) == "toolCall"]
    assert len(tcs) == 1
    assert tcs[0].name == "grep"
    assert tcs[0].arguments == {"pattern": "TODO"}


@pytest.mark.asyncio
async def test_reasoning_mapping():
    model = _model("deepseek")
    out = AssistantMessage(role="assistant", content=[], api=model.api, provider=model.provider, model=model.id)
    stream = AssistantMessageEventStream(loop=asyncio.get_event_loop())
    await _feed(
        [
            {"choices": [{"delta": {"reasoning_content": "let me think"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}]},
        ],
        model,
        out,
        stream,
    )
    thinking = [b for b in out.content if getattr(b, "type", None) == "thinking"]
    texts = [b for b in out.content if getattr(b, "type", None) == "text"]
    assert thinking and thinking[0].thinking == "let me think"
    assert texts and texts[0].text == "answer"


@pytest.mark.asyncio
async def test_usage_mapping():
    model = _model("openrouter")
    from pi_ai.models import calculate_cost

    out = AssistantMessage(role="assistant", content=[], api=model.api, provider=model.provider, model=model.id)
    stream = AssistantMessageEventStream(loop=asyncio.get_event_loop())
    await _feed(
        [
            {
                "choices": [{"delta": {}, "finish_reason": None}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "prompt_tokens_details": {"cached_tokens": 2},
                    "completion_tokens_details": {"reasoning_tokens": 1},
                },
            },
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ],
        model,
        out,
        stream,
    )
    assert out.usage.input == 8
    assert out.usage.output == 5
    assert out.usage.cacheRead == 2
    assert out.usage.reasoning == 1


def test_detect_compat_thinking_format():
    assert detect_compat(_model("openrouter")).thinking_format == "openrouter"
    assert detect_compat(_model("deepseek")).thinking_format == "deepseek"
    assert detect_compat(_model("qwen")).thinking_format == "qwen"
    assert detect_compat(_model("openai")).thinking_format == "openai"
    # deepseek flag
    assert detect_compat(_model("deepseek")).requires_reasoning_content_on_assistant is True


def test_map_stop_reason():
    assert map_stop_reason("stop") == ("stop", None)
    assert map_stop_reason("length") == ("length", None)
    assert map_stop_reason("tool_calls") == ("toolUse", None)
    sr, msg = map_stop_reason("content_filter")
    assert sr == "error" and msg is not None
