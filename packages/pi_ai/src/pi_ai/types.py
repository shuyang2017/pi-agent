"""Core data types for pi-ai (Python port of packages/ai/src/types.ts).

Faithful port of the upstream TypeScript type definitions. Discriminated unions
(AssistantMessageEvent, AgentEvent) are modeled as a single dataclass with a
``type`` discriminator and optional fields, which is the natural Python
equivalent of TypeScript's `type: "x" | "y"` union and keeps ``event.type``
checks identical to the upstream source.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

if sys.version_info >= (3, 12):
    from typing import TypedDict
else:  # pragma: no cover
    from typing_extensions import TypedDict

# ---------------------------------------------------------------------------
# Stop reasons (mirrors upstream StopReason union)
# ---------------------------------------------------------------------------
StopReason = Literal[
    "pending",
    "stop",
    "length",
    "toolUse",
    "error",
    "aborted",
    "deferred",
]


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------
@dataclass
class TextContent:
    type: Literal["text"] = "text"
    text: str = ""
    textSignature: Optional[str] = None  # OpenAI responses metadata (legacy id or TextSignatureV1 JSON)


@dataclass
class ThinkingContent:
    type: Literal["thinking"] = "thinking"
    thinking: str = ""
    thinkingSignature: Optional[str] = None
    redacted: bool = False


@dataclass
class ImageContent:
    type: Literal["image"] = "image"
    data: str = ""  # base64
    mimeType: str = "image/png"


@dataclass
class ToolCall:
    type: Literal["toolCall"] = "toolCall"
    id: str = ""
    name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    thoughtSignature: Optional[str] = None
    namespace: Optional[str] = None


ContentBlock = Union[TextContent, ThinkingContent, ToolCall]


@dataclass
class ToolResultContent:
    type: Literal["text", "image"] = "text"
    text: str = ""
    mimeType: Optional[str] = None
    data: Optional[str] = None


# ---------------------------------------------------------------------------
# Usage + cost
# ---------------------------------------------------------------------------
@dataclass
class ModelCostRates:
    input: float = 0.0  # $/million tokens
    output: float = 0.0
    cacheRead: float = 0.0
    cacheWrite: float = 0.0


@dataclass
class ModelCost(ModelCostRates):
    tiers: Optional[List["ModelCostTier"]] = None


@dataclass
class ModelCostTier(ModelCostRates):
    inputTokensAbove: int = 0


@dataclass
class Usage:
    input: int = 0
    output: int = 0
    cacheRead: int = 0
    cacheWrite: int = 0
    cacheWrite1h: int = 0
    reasoning: int = 0
    totalTokens: int = 0
    cost: ModelCostRates = field(default_factory=ModelCostRates)


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------
@dataclass
class UserMessage:
    role: Literal["user"] = "user"
    content: Union[str, List[Union[TextContent, ImageContent]]] = ""
    timestamp: int = 0


@dataclass
class AssistantMessage:
    role: Literal["assistant"] = "assistant"
    content: List[ContentBlock] = field(default_factory=list)
    api: str = ""
    provider: str = ""
    model: str = ""
    responseModel: Optional[str] = None
    responseId: Optional[str] = None
    usage: Usage = field(default_factory=Usage)
    stopReason: StopReason = "pending"  # type: ignore[assignment]
    deferred: Optional[Any] = None
    errorMessage: Optional[str] = None
    rawStopReason: Optional[str] = None
    endTurn: Optional[bool] = None
    timestamp: int = 0


@dataclass
class ToolResultMessage:
    role: Literal["toolResult"] = "toolResult"
    toolCallId: str = ""
    toolName: str = ""
    content: List[Union[TextContent, ImageContent]] = field(default_factory=list)
    details: Any = None
    usage: Optional[Usage] = None
    addedToolNames: Optional[List[str]] = None
    isError: bool = False
    timestamp: int = 0


Message = Union[UserMessage, AssistantMessage, ToolResultMessage]


# ---------------------------------------------------------------------------
# Tool + Model + Context
# ---------------------------------------------------------------------------
@dataclass
class Tool:
    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)  # JSON schema
    constrainedSampling: Any = None


@dataclass
class Model:
    id: str = ""
    name: str = ""
    api: str = ""
    provider: str = ""
    baseUrl: str = ""
    reasoning: bool = False
    thinkingLevelMap: Optional[Dict[str, Optional[str]]] = None
    input: List[str] = field(default_factory=lambda: ["text"])
    cost: ModelCost = field(default_factory=ModelCost)
    contextWindow: int = 200_000
    maxTokens: int = 8_192
    samplingParams: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, str]] = None
    compat: Optional[Dict[str, Any]] = None


@dataclass
class Context:
    messages: List[Message] = field(default_factory=list)
    systemPrompt: Optional[str] = None
    tools: Optional[List[Tool]] = None


# ---------------------------------------------------------------------------
# AssistantMessageEvent — single dataclass with `type` discriminator.
# Mirrors the upstream discriminated union; only the fields relevant to the
# current `type` are populated.
# ---------------------------------------------------------------------------
@dataclass
class AssistantMessageEvent:
    type: str
    partial: AssistantMessage
    contentIndex: Optional[int] = None
    delta: Optional[str] = None
    content: Optional[str] = None
    toolCall: Optional[ToolCall] = None
    reason: Optional[str] = None  # "stop" | "length" | "toolUse" | "deferred" | "aborted" | "error"
    error: Optional[AssistantMessage] = None


# StreamFn contract: returns an AssistantMessageEventStream (or awaitable of one).
StreamFn = Any


def create_empty_usage() -> Usage:
    return Usage()


__all__ = [
    "StopReason",
    "TextContent",
    "ThinkingContent",
    "ImageContent",
    "ToolCall",
    "ContentBlock",
    "ToolResultContent",
    "ModelCostRates",
    "ModelCost",
    "ModelCostTier",
    "Usage",
    "UserMessage",
    "AssistantMessage",
    "ToolResultMessage",
    "Message",
    "Tool",
    "Model",
    "Context",
    "AssistantMessageEvent",
    "StreamFn",
    "create_empty_usage",
]
