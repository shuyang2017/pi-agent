"""Streaming option types (Python port of packages/ai/src/types.ts StreamOptions / SimpleStreamOptions).

Kept dependency-free so both pi-ai adapters and pi-agent-core's AgentLoopConfig
can reuse them (upstream: ``AgentLoopConfig extends SimpleStreamOptions``).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class StreamOptions:
    apiKey: Optional[str] = None
    headers: Optional[Dict[str, Optional[str]]] = None
    maxTokens: Optional[int] = None
    temperature: Optional[float] = None
    reasoning: Optional[str] = None  # ThinkingLevel: off|minimal|low|medium|high|xhigh|max
    cacheRetention: str = "short"
    sessionId: Optional[str] = None
    signal: Optional[asyncio.Event] = None
    timeout: Optional[float] = None
    maxRetries: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    toolChoice: Optional[Any] = None
    env: Optional[Dict[str, str]] = None
    onResponse: Optional[Callable[..., Any]] = None
    onPayload: Optional[Callable[..., Any]] = None


@dataclass
class SimpleStreamOptions(StreamOptions):
    thinkingBudgets: Optional[Dict[str, int]] = None


@dataclass
class AnthropicOptions(SimpleStreamOptions):
    thinkingEnabled: Optional[bool] = None
    thinkingBudgetTokens: Optional[int] = None
    thinkingDisplay: str = "summarized"
    interleavedThinking: bool = True
    effort: Optional[str] = None
    client: Optional[Any] = None


# Google thinking config (subset of upstream GoogleOptions.thinking).
@dataclass
class GoogleThinkingConfig:
    enabled: bool = False
    budgetTokens: Optional[int] = None  # -1 dynamic, 0 disable
    level: Optional[str] = None  # GoogleThinkingLevel (MINIMAL|LOW|MEDIUM|HIGH|...)


@dataclass
class GoogleOptions(SimpleStreamOptions):
    thinking: Optional[GoogleThinkingConfig] = None
    toolChoice: Optional[Any] = None  # auto | none | any


@dataclass
class BedrockOptions(SimpleStreamOptions):
    profile: Optional[str] = None
    bearerToken: Optional[str] = None
    reasoning: Optional[Any] = None
    thinkingBudgets: Optional[Dict[str, int]] = None


__all__ = [
    "StreamOptions",
    "SimpleStreamOptions",
    "AnthropicOptions",
    "GoogleOptions",
    "GoogleThinkingConfig",
    "BedrockOptions",
]
