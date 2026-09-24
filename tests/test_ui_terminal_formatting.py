import io

from rich.console import Console

from ui.rich_terminal import RichTerminalUI


def _render(text: str) -> str:
    ui = RichTerminalUI(user_name="Tester")
    ui.console = Console(file=io.StringIO(), width=400, color_system=None)
    ui.print_assistant_response(text)
    return ui.console.file.getvalue()


def test_reply_is_not_boxed_or_timestamped():
    out = _render("I can map a practical AI build path from here.\nWhich part first?")

    assert "╭" not in out and "│" not in out
    assert "A.L.I.C.E" not in out
    assert "I can map a practical AI build path from here." in out


def test_long_single_paragraph_is_not_split_into_sentences():
    text = (
        "I can map a practical AI build path from here. "
        "A solid next set of tracks is intent routing, state handling, and verification strategy. "
        "Which direction should we prioritize first?"
    )

    out = _render(text)

    assert text in out.splitlines()


def test_section_words_in_prose_do_not_break_sentences():
    text = (
        "I looked at your notes from last night. The Approach: you wanted to batch the writes, "
        "which still seems right. The Timeline: two weeks."
    )

    out = _render(text)

    assert text in out.splitlines()


def test_dunder_filenames_survive_markdown_rendering():
    out = _render("Two files changed:\n- ai/memory/__init__.py\n- `app/__main__.py`")

    assert "__init__.py" in out
    assert "__main__.py" in out


def test_markdown_lists_still_render_as_lists():
    out = _render("## Plan\n- First step\n- Second step")

    assert "##" not in out
    assert "First step" in out and "Second step" in out
