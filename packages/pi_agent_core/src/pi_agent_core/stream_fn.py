"""Default stream-fn registry (Python port of packages/agent/src/stream-fn.ts)."""

from __future__ import annotations

from typing import Any, Optional

from .types import StreamFn

_default_stream_fn: Optional[StreamFn] = None


def set_default_stream_fn(stream_fn: Optional[StreamFn]) -> None:
    global _default_stream_fn
    _default_stream_fn = stream_fn


def get_default_stream_fn() -> StreamFn:
    if _default_stream_fn is not None:
        return _default_stream_fn
    # Lazy fallback: route to pi-ai's provider dispatcher (no explicit wiring needed).
    from pi_ai.stream import stream_simple

    return stream_simple


__all__ = ["set_default_stream_fn", "get_default_stream_fn"]
