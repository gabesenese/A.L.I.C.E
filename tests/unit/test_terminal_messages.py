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
