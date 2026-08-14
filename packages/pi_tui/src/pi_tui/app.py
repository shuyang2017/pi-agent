"""Interactive Textual UI for the coding agent.

Subscribes to :meth:`AgentSession.stream_turn` and renders the same event
vocabulary used by the print / rpc front-ends — agent text streams inline,
tool calls are shown as bracketed lines, and the committed history survives
each turn.

The streaming assistant text is rendered into a live ``Static`` widget that is
rewritten on every ``message_update`` delta (true token streaming), then
committed to the ``RichLog`` history when the assistant turn ends.
"""

from __future__ import annotations

from typing import Optional

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Footer, Header, Input, RichLog, Static

from pi_agent_core.types import AgentEvent
from pi_coding_agent.agent_session import AgentSession


class CodingAgentApp(App):
    """Textual chat client driving a :class:`AgentSession`."""

    CSS = """
    #chat {
        height: 1fr;
        border: round $panel;
        padding: 0 1;
    }
    #live {
        height: auto;
        max-height: 12;
        padding: 0 1;
        color: $text;
    }
    #prompt {
        dock: bottom;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(
        self,
        session: AgentSession,
        title: str = "pi coding agent",
    ) -> None:
        super().__init__()
        self.session = session
        self.title = title
        self._busy = False
        self._live = ""

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield RichLog(id="chat", wrap=True, markup=False)
            yield Static(id="live", markup=False)
        yield Input(placeholder="Ask the agent… (Ctrl+C to quit)", id="prompt")
        yield Footer()

    async def on_mount(self) -> None:
        log = self.query_one("#chat", RichLog)
        log.write("pi coding-agent ready. Type a request and press Enter.")
        self.query_one("#prompt", Input).focus()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""
        if self._busy:
            self.query_one("#chat", RichLog).write(
                "[ignored: a turn is already running]"
            )
            return
        await self._run_turn(text)

    async def _run_turn(self, text: str) -> None:
        self._busy = True
        self.query_one("#prompt", Input).disabled = True
        log = self.query_one("#chat", RichLog)
        log.write(f"[you] {text}")
        try:
            await self._stream_turn(text, log)
        except Exception as error:  # surface failures instead of crashing the UI
            log.write(f"[error] {error}")
        finally:
            self._commit_live(log)
            self._busy = False
            self.query_one("#prompt", Input).disabled = False
            self.query_one("#prompt", Input).focus()

    async def _stream_turn(self, text: str, log: RichLog) -> None:
        async for ev in self.session.stream_turn(text):
            self._render_event(ev, log)

    def _render_event(self, ev: AgentEvent, log: RichLog) -> None:
        t = ev.type
        if t == "message_update" and ev.assistantMessageEvent is not None:
            delta = getattr(ev.assistantMessageEvent, "delta", None)
            if delta:
                self._live += delta
                self.query_one("#live", Static).update(self._live)
        elif t == "tool_execution_start":
            self._commit_live(log)
            args = ev.args if isinstance(ev.args, (str, int, float, bool)) else ev.args
            log.write(f"[tool {ev.toolName}] {args}")
        elif t == "tool_execution_end":
            log.write("[tool done]" if not ev.isError else "[tool error]")
        elif t in ("message_end", "turn_end"):
            self._commit_live(log)

    def _commit_live(self, log: RichLog) -> None:
        if self._live:
            log.write(self._live)
            self._live = ""
            self.query_one("#live", Static).update("")
