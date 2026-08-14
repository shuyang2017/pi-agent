"""Interactive Textual app test: drive a real turn via the pilot harness."""

from textual.widgets import Input, RichLog

from pi_tui.app import CodingAgentApp
from pi_tui.session_builder import build_session_from_env


async def test_interactive_end_to_end_renders_and_updates_history():
    session = build_session_from_env(mock=True)
    app = CodingAgentApp(session)

    async with app.run_test() as pil:
        captured = []
        chat = app.query_one(RichLog)
        original_write = chat.write

        def capturing_write(*args, **kwargs):
            captured.append(args[0] if args else None)
            return original_write(*args, **kwargs)

        chat.write = capturing_write  # type: ignore[assignment]

        inp = app.query_one(Input)
        inp.value = "hello"
        app.post_message(inp.Submitted(input=inp, value="hello"))

        # Let the agent turn run to completion (user + assistant messages landed).
        for _ in range(60):
            await pil.pause(0.05)
            if any(getattr(m, "role", None) == "assistant" for m in session.messages):
                break

    joined = "\n".join(str(c) for c in captured if c is not None)
    assert "(mock) Echo: hello" in joined

    assistant_texts = [
        getattr(m.content[0], "text", "")
        for m in session.messages
        if getattr(m, "role", None) == "assistant" and getattr(m, "content", None)
    ]
    assert any("(mock) Echo: hello" in t for t in assistant_texts)
