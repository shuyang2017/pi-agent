"""pi-ai API adapters (per-provider streaming implementations)."""

from __future__ import annotations

from . import bedrock, google, openrouter, qwen, deepseek
from ._openai_compat import stream as openai_completions_stream, stream_simple as openai_completions_stream_simple
from .anthropic_messages import stream as anthropic_stream, stream_simple as anthropic_stream_simple
from .bedrock import stream as bedrock_stream, stream_simple as bedrock_stream_simple
from .google import stream as google_stream, stream_simple as google_stream_simple
from .openrouter import stream as openrouter_stream, stream_simple as openrouter_stream_simple
from .qwen import stream as qwen_stream, stream_simple as qwen_stream_simple
from .deepseek import stream as deepseek_stream, stream_simple as deepseek_stream_simple

__all__ = [
    "anthropic_messages",
    "google",
    "bedrock",
    "openrouter",
    "qwen",
    "deepseek",
    "openai_completions_stream",
    "openai_completions_stream_simple",
    "anthropic_stream",
    "anthropic_stream_simple",
    "google_stream",
    "google_stream_simple",
    "bedrock_stream",
    "bedrock_stream_simple",
    "openrouter_stream",
    "openrouter_stream_simple",
    "qwen_stream",
    "qwen_stream_simple",
    "deepseek_stream",
    "deepseek_stream_simple",
]
