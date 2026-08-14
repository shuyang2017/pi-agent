"""EventStream: generic async event stream (Python port of packages/ai/src/utils/event-stream.ts).

The upstream class pairs a queue with a list of waiting resolvers to implement a
single-consumer rendezvous:

    push(event):
      if done: return
      if isComplete(event):
        done = true; resolveFinalResult(extractResult(event))
      waiter = waiting.shift()
      if waiter: waiter({value: event, done: false})
      else: queue.push(event)

    end(result?):
      done = true
      if result !== undefined: resolveFinalResult(result)
      while waiting: waiting.shift()({value: undefined, done: true})

    async iterator:
      loop:
        if queue: yield queue.shift()
        elif done: return
        else: await new Promise(resolve => waiting.push(resolve))  # then yield value

Python maps the Promise rendezvous onto asyncio.Future and the AsyncIterable onto
an async generator. The final result is carried by a separate Future
(``result()``), exactly like the upstream ``.result()``.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Generic, Optional, TypeVar, Union

from .types import AssistantMessage, AssistantMessageEvent

T = TypeVar("T")
R = TypeVar("R")


def _resolve_loop(loop: Optional[asyncio.AbstractEventLoop]) -> asyncio.AbstractEventLoop:
    """Return ``loop`` or a usable event loop.

    ``asyncio.get_event_loop()`` raises ``RuntimeError`` (3.10+) when called
    outside a running loop *after* a prior loop was closed and unset (exactly
    what pytest-asyncio does between tests). Fall back to a fresh loop so the
    stream can always be constructed.
    """
    if loop is not None:
        return loop
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop


class _IteratorResult:
    __slots__ = ("value", "done")

    def __init__(self, value: Any, done: bool):
        self.value = value
        self.done = done


class EventStream(Generic[T, R]):
    def __init__(
        self,
        is_complete: Callable[[T], bool],
        extract_result: Callable[[T], R],
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self._queue: list[T] = []
        self._waiting: list[asyncio.Future] = []
        self._done = False
        self._is_complete = is_complete
        self._extract_result = extract_result
        self._loop = _resolve_loop(loop)
        self._result_future: asyncio.Future = self._loop.create_future()

    # -- producer side ----------------------------------------------------
    def push(self, event: T) -> None:
        if self._done:
            return

        if self._is_complete(event):
            self._done = True
            if not self._result_future.done():
                self._result_future.set_result(self._extract_result(event))

        if self._waiting:
            waiter = self._waiting.pop(0)
            if not waiter.done():
                waiter.set_result(_IteratorResult(event, False))
        else:
            self._queue.append(event)

    def end(self, result: Optional[R] = None) -> None:
        self._done = True
        if result is not None and not self._result_future.done():
            self._result_future.set_result(result)
        while self._waiting:
            waiter = self._waiting.pop(0)
            if not waiter.done():
                waiter.set_result(_IteratorResult(None, True))

    # -- consumer side ----------------------------------------------------
    def __aiter__(self):
        return self._aiter()

    async def _aiter(self):
        while True:
            if self._queue:
                yield self._queue.pop(0)
            elif self._done:
                return
            else:
                fut: asyncio.Future = asyncio.get_event_loop().create_future()
                self._waiting.append(fut)
                result = await fut
                if result.done:
                    return
                yield result.value

    async def result(self) -> R:
        return await self._result_future


class AssistantMessageEventStream(EventStream[AssistantMessageEvent, AssistantMessage]):
    """Specializes EventStream for assistant-message streams.

    Terminates on ``done`` / ``error`` events; the final result is the
    assistant message carried by that terminal event (for ``error`` the message
    has stopReason ``"error"`` / ``"aborted"`` and an errorMessage).
    """

    def __init__(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        super().__init__(
            is_complete=lambda e: e.type in ("done", "error"),
            extract_result=self._extract,
            loop=loop,
        )

    @staticmethod
    def _extract(event: AssistantMessageEvent) -> AssistantMessage:
        if event.type == "done":
            return event.partial  # `partial` holds the finalized message on done
        if event.type == "error":
            return event.error if event.error is not None else event.partial
        raise ValueError("Unexpected terminal event type for result")


def create_assistant_message_event_stream(
    loop: Optional[asyncio.AbstractEventLoop] = None,
) -> AssistantMessageEventStream:
    return AssistantMessageEventStream(loop=loop)


__all__ = [
    "EventStream",
    "AssistantMessageEventStream",
    "create_assistant_message_event_stream",
]
