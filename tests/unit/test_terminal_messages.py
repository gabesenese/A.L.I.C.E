"""Slash commands and errors read as sentences, not log lines."""

from types import SimpleNamespace

from app.main import ALICE
from ui.rich_terminal import RichTerminalUI


def test_slash_command_confirmation_has_no_status_tag(capsys):
    alice = ALICE.__new__(ALICE)
    alice.llm = SimpleNamespace(clear_history=lambda: None)
    alice.context = SimpleNamespace(clear_short_term_memory=lambda: None)

    alice._handle_command("/clear")
    alice._handle_command("/nonsense")

    out = capsys.readouterr().out
    assert "Conversation history cleared" in out
    assert "Unknown command: /nonsense" in out
    assert "[OK]" not in out and "[ERROR]" not in out


def test_error_line_has_no_error_label():
    ui = RichTerminalUI()
    with ui.console.capture() as captured:
        ui.print_error("Something broke on my side while handling that.")

    assert captured.get().strip() == "Something broke on my side while handling that."


def test_spinner_shows_elapsed_seconds_once_the_turn_is_slow():
    from ui.rich_terminal import ThinkingStatus

    now = [100.0]
    status = ThinkingStatus(clock=lambda: now[0])
    assert status.label() == "thinking…"

    now[0] = 112.4
    assert status.label() == "thinking… 12s"

    status.set_phase("reading agent_loop.py…")
    assert status.label() == "reading agent_loop.py… 12s"


def test_spinner_renders_inside_a_live_display():
    from io import StringIO

    from rich.console import Console

    ui = RichTerminalUI.__new__(RichTerminalUI)
    ui.console = Console(file=StringIO(), force_terminal=True)
    ui.colors = {"info": "cyan"}
    with ui.thinking_spinner() as status:
        assert status.label().startswith("thinking")
