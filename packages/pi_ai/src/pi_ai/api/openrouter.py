"""OpenRouter adapter (Python port of providers/openrouter.ts).

OpenRouter is an OpenAI-compatible endpoint; the streaming engine lives in
``_openai_compat``. This module is a thin, well-named facade that routes to the
shared engine. Construct models with ``provider="openrouter"`` (and optionally
``baseUrl``); ``detect_compat`` then selects the ``openrouter`` reasoning format.
"""

from __future__ import annotations

from typing import Optional

from ..event_stream import AssistantMessageEventStream
from ..options import SimpleStreamOptions, StreamOptions
from ..types import Context, Model
from ._openai_compat import stream as _stream, stream_simple as _stream_simple

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def stream(
    model: Model,
    context: Context,
    options: Optional[StreamOptions] = None,
) -> AssistantMessageEventStream:
    return _stream(model, context, options)


def stream_simple(
    model: Model,
    context: Context,
    options: Optional[SimpleStreamOptions] = None,
) -> AssistantMessageEventStream:
    return _stream_simple(model, context, options)


__all__ = ["stream", "stream_simple", "DEFAULT_BASE_URL"]
