"""AgentSessionRuntime + JSONL session store (Python port of coding-agent core).

The upstream runtime is wired to a heavyweight ``SessionManager`` (JSONL
history, branching, extensions, model registry). For this faithful vertical
slice we keep the same *surface* — ``new_session`` / ``resume`` / ``fork`` /
``import_from_jsonl`` / ``dispose`` — backed by a small in-memory +
JSONL session store. Production isolation (Docker sandbox) is out of scope
here and noted where relevant.

Lifecycle rules mirror upstream teardown: settle any active turn first (set the
abort signal), then replace the conversation.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from typing import Any, List, Optional

from pi_ai import AssistantMessage, Model, TextContent, ThinkingContent, ToolCall, ToolResultMessage, Usage, UserMessage
from pi_ai.types import ImageContent

from .agent_session import AgentSession

__all__ = ["AgentSessionRuntime", "save_messages", "load_messages", "SessionStore"]


# ---------------------------------------------------------------------------
# Message (de)serialization for JSONL
# ---------------------------------------------------------------------------
_CONTENT_TYPES = {
    "text": TextContent,
    "thinking": ThinkingContent,
    "toolCall": ToolCall,
    "image": ImageContent,
}


def _rebuild_content(items: Any) -> List[Any]:
    out: List[Any] = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        cls = _CONTENT_TYPES.get(it.get("type"))
        if cls is None:
            continue
        out.append(cls(**{k: v for k, v in it.items() if k != "type"}))
    return out


def _rebuild_usage(d: Any) -> Usage:
    if not isinstance(d, dict):
        return Usage()
    return Usage(
        input=d.get("input", 0),
        output=d.get("output", 0),
        cacheRead=d.get("cacheRead", 0),
        cacheWrite=d.get("cacheWrite", 0),
        cacheWrite1h=d.get("cacheWrite1h", 0),
        reasoning=d.get("reasoning", 0),
        totalTokens=d.get("totalTokens", 0),
    )


def _build_message(role: str, d: dict) -> Any:
    if role == "user":
        content = d.get("content")
        if isinstance(content, list):
            content = _rebuild_content(content)
        return UserMessage(role="user", content=content, timestamp=d.get("timestamp", 0))
    if role == "assistant":
        return AssistantMessage(
            role="assistant",
            content=_rebuild_content(d.get("content", [])),
            api=d.get("api", ""),
            provider=d.get("provider", ""),
            model=d.get("model", ""),
            usage=_rebuild_usage(d.get("usage")),
            stopReason=d.get("stopReason", "stop"),
            errorMessage=d.get("errorMessage"),
            timestamp=d.get("timestamp", 0),
        )
    if role == "toolResult":
        return ToolResultMessage(
            role="toolResult",
            toolCallId=d.get("toolCallId", ""),
            toolName=d.get("toolName", ""),
            content=_rebuild_content(d.get("content", [])),
            isError=d.get("isError", False),
            timestamp=d.get("timestamp", 0),
        )
    raise ValueError(f"Unknown message role: {role!r}")


def save_messages(messages: List[Any], path: str) -> None:
    """Serialize a conversation to a JSONL file (one JSON object per line)."""
    rows = []
    for m in messages:
        d = dataclasses.asdict(m)
        d["role"] = m.role
        rows.append(d)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)


def load_messages(path: str) -> List[Any]:
    """Load a conversation from a JSONL file produced by :func:`save_messages`."""
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    return [_build_message(r.get("role"), r) for r in rows]


class SessionStore:
    """Thin JSONL-backed persistence wrapper."""

    def __init__(self, directory: str) -> None:
        self.directory = directory

    def path_for(self, name: str) -> str:
        import os

        return os.path.join(self.directory, f"{name}.jsonl")

    def save(self, name: str, messages: List[Any]) -> str:
        import os

        os.makedirs(self.directory, exist_ok=True)
        path = self.path_for(name)
        save_messages(messages, path)
        return path

    def load(self, name: str) -> List[Any]:
        return load_messages(self.path_for(name))


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
@dataclass
class AgentSessionRuntime:
    """Owns an :class:`AgentSession` and its lifecycle commands."""

    session: AgentSession

    @property
    def messages(self) -> List[Any]:
        return self.session.messages

    async def _teardown(self) -> None:
        # Settle any active turn before replacing the conversation. The session
        # holds the abort signal; agent_loop checks it at tool-dispatch points.
        self.session.abort()

    async def new_session(self) -> None:
        """Start a fresh conversation (clears history)."""
        await self._teardown()
        self.session.messages = []

    async def resume(self, path: str) -> None:
        """Replace history by loading a previously saved session file."""
        await self._teardown()
        self.session.messages = load_messages(path)

    async def fork(self, entry_id: str) -> None:
        """Branch the conversation: keep only the prefix before ``entry_id``.

        ``entry_id`` is interpreted as an index into the current message list
        (the upstream uses a tree entry id; for the slice we key by index).
        """
        await self._teardown()
        index = int(entry_id) if str(entry_id).isdigit() else 0
        self.session.messages = self.session.messages[: max(0, index)]

    async def import_from_jsonl(self, path: str) -> None:
        """Import a session JSONL exported elsewhere and switch to it."""
        await self._teardown()
        self.session.messages = load_messages(path)

    async def dispose(self) -> None:
        """Tear down the runtime (abort active turn)."""
        await self._teardown()
