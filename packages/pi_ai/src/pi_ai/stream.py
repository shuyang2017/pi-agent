"""Provider dispatch (Python port of the upstream ``ai`` stream entry point).

A single ``stream`` / ``stream_simple`` keyed by ``model.api`` selects the right
per-provider adapter. Agent runtimes wire this as their ``stream_fn`` (see
``pi_agent_core.stream_fn``).
"""

from __future__ import annotations

from typing import Optional

from .api import (
    anthropic_stream,
    anthropic_stream_simple,
    bedrock_stream,
    bedrock_stream_simple,
    google_stream,
    google_stream_simple,
    openai_completions_stream,
    openai_completions_stream_simple,
)
from .event_stream import AssistantMessageEventStream
from .options import SimpleStreamOptions, StreamOptions
from .types import Context, Model

_API_STREAMS = {
    "anthropic-messages": (anthropic_stream, anthropic_stream_simple),
    "google-generative-ai": (google_stream, google_stream_simple),
    "bedrock-converse-stream": (bedrock_stream, bedrock_stream_simple),
    "openai-completions": (openai_completions_stream, openai_completions_stream_simple),
}


def stream(
    model: Model,
    context: Context,
    options: Optional[StreamOptions] = None,
) -> AssistantMessageEventStream:
    pair = _API_STREAMS.get(model.api)
    if pair is None:
        raise ValueError(
            f"Unknown model.api: {model.api!r}. Supported: {sorted(_API_STREAMS)}"
        )
    return pair[0](model, context, options)


def stream_simple(
    model: Model,
    context: Context,
    options: Optional[SimpleStreamOptions] = None,
) -> AssistantMessageEventStream:
    pair = _API_STREAMS.get(model.api)
    if pair is None:
        raise ValueError(
            f"Unknown model.api: {model.api!r}. Supported: {sorted(_API_STREAMS)}"
        )
    return pair[1](model, context, options)


__all__ = ["stream", "stream_simple"]
