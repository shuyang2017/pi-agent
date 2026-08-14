"""Agent runtime types (Python port of packages/agent/src/types.ts).

AgentMessage reuses pi-ai's Message union. AgentEvent is modeled as a single
dataclass with a ``type`` discriminator (same approach as AssistantMessageEvent).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from pi_ai import AssistantMessage, AssistantMessageEvent, Context, Model, Tool, ToolCall, ToolResultMessage, Usage, UserMessage

AgentMessage = Union[UserMessage, AssistantMessage, ToolResultMessage]
Message = AgentMessage  # pi-ai Message alias for convertToLlm return
AgentToolCall = ToolCall

StreamFn = Callable[..., Any]

# Tool dispatch strategy (mirrors upstream `type ToolExecutionMode = "sequential" | "parallel"`).
ToolExecutionMode = Literal["sequential", "parallel"]


@dataclass
class AgentToolResult:
    content: List[Union[TextContent, Any]] = field(default_factory=list)
    details: Any = None
    usage: Optional[Usage] = None
    addedToolNames: Optional[List[str]] = None
    terminate: bool = False


# Re-use pi-ai Tool but add runtime fields (label, execute, ...).
@dataclass
class AgentTool(Tool):
    label: str = ""
    execute: Optional[Callable[..., Any]] = None
    prepareArguments: Optional[Callable[[Any], Any]] = None
    executionMode: Optional[str] = None  # "sequential" | "parallel"


# AgentToolResult needs TextContent; import here to avoid cycle.
from pi_ai import TextContent  # noqa: E402


@dataclass
class AgentContext:
    systemPrompt: str = ""
    messages: List[AgentMessage] = field(default_factory=list)
    tools: Optional[List[AgentTool]] = None


# ---------------------------------------------------------------------------
# AgentEvent: single dataclass, `type` discriminator.
# ---------------------------------------------------------------------------
@dataclass
class AgentEvent:
    type: str
    messages: Optional[List[AgentMessage]] = None
    message: Optional[AgentMessage] = None
    toolResults: Optional[List[ToolResultMessage]] = None
    assistantMessageEvent: Optional[AssistantMessageEvent] = None
    toolCallId: Optional[str] = None
    toolName: Optional[str] = None
    args: Any = None
    partialResult: Any = None
    result: Any = None
    isError: bool = False


# ---------------------------------------------------------------------------
# Hook result types
# ---------------------------------------------------------------------------
@dataclass
class BeforeToolCallResult:
    block: bool = False
    reason: Optional[str] = None
    terminate: bool = False


@dataclass
class AfterToolCallResult:
    content: Optional[List[Any]] = None
    details: Any = None
    isError: Optional[bool] = None
    usage: Optional[Usage] = None
    terminate: Optional[bool] = None


@dataclass
class BeforeToolCallContext:
    assistantMessage: AssistantMessage
    toolCall: AgentToolCall
    args: Any
    context: AgentContext


@dataclass
class AfterToolCallContext:
    assistantMessage: AssistantMessage
    toolCall: AgentToolCall
    args: Any
    result: AgentToolResult
    isError: bool
    context: AgentContext


@dataclass
class ShouldStopAfterTurnContext:
    message: AssistantMessage
    toolResults: List[ToolResultMessage]
    context: AgentContext
    newMessages: List[AgentMessage]


@dataclass
class AgentLoopTurnUpdate:
    context: Optional[AgentContext] = None
    model: Optional[Model] = None
    thinkingLevel: Optional[str] = None


@dataclass
class AgentLoopConfig:
    # --- required ---
    model: Model
    convertToLlm: Callable[[List[AgentMessage]], Any]

    # --- streaming option fields (mirrors SimpleStreamOptions + AnthropicOptions) ---
    apiKey: Optional[str] = None
    headers: Optional[Dict[str, Optional[str]]] = None
    maxTokens: Optional[int] = None
    temperature: Optional[float] = None
    reasoning: Optional[str] = None
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
    thinkingBudgets: Optional[Dict[str, int]] = None
    thinkingEnabled: Optional[bool] = None
    thinkingBudgetTokens: Optional[int] = None
    thinkingDisplay: str = "summarized"
    interleavedThinking: bool = True
    effort: Optional[str] = None

    # --- agent-loop hooks ---
    transformContext: Optional[Callable[[List[AgentMessage], Optional[asyncio.Event]], Any]] = None
    getApiKey: Optional[Callable[[str], Any]] = None
    shouldStopAfterTurn: Optional[Callable[[ShouldStopAfterTurnContext], Any]] = None
    prepareNextTurn: Optional[Callable[[Any], Any]] = None
    getSteeringMessages: Optional[Callable[[], Any]] = None
    getFollowUpMessages: Optional[Callable[[], Any]] = None
    toolExecution: Optional[str] = "parallel"
    beforeToolCall: Optional[Callable[[BeforeToolCallContext, Optional[asyncio.Event]], Any]] = None
    afterToolCall: Optional[Callable[[AfterToolCallContext, Optional[asyncio.Event]], Any]] = None


__all__ = [
    "AgentMessage",
    "Message",
    "AgentToolCall",
    "AgentTool",
    "AgentToolResult",
    "AgentContext",
    "AgentEvent",
    "BeforeToolCallResult",
    "AfterToolCallResult",
    "BeforeToolCallContext",
    "AfterToolCallContext",
    "ShouldStopAfterTurnContext",
    "AgentLoopTurnUpdate",
    "AgentLoopConfig",
    "StreamFn",
    "ToolExecutionMode",
]
