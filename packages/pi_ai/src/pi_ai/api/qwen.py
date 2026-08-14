"""Qwen (DashScope) adapter (Python port of providers/qwen-token-plan*.ts).

Qwen's compatible-mode endpoint is OpenAI-compatible; the streaming engine lives
in ``_openai_compat``. Use ``provider="qwen"`` so ``detect_compat`` selects the
``qwen`` reasoning format (``enable_thinking`` / ``reasoning_effort``).
"""

from __future__ import annotations

from typing import Optional

from ..event_stream import AssistantMessageEventStream
from ..options import SimpleStreamOptions, StreamOptions
from ..types import Context, Model
from ._openai_compat import stream as _stream, stream_simple as _stream_simple

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


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
