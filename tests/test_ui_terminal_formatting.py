from ui.rich_terminal import RichTerminalUI


def test_long_plain_response_gets_sentence_spacing_for_terminal_readability():
    ui = RichTerminalUI(user_name="Tester")
    text = (
        "Good. AI is a strong place to start. You could build a retrieval-augmented Q&A app, "
        "a multi-step reasoning assistant, an agent with tool-use, a summarization copilot, "
        "and a domain tutor. Do you want to focus first on memory, tool-use, or conversational quality?"
    )

    out = ui._format_assistant_terminal_text(text)

    assert "\n\n" in out
    assert out.startswith("Good. AI is a strong place to start.")


def test_short_plain_response_keeps_single_line_format():
    ui = RichTerminalUI(user_name="Tester")
    text = "Thanks. I can help with that."

    out = ui._format_assistant_terminal_text(text)

    assert out == text


def test_markdown_response_is_not_reformatted():
    ui = RichTerminalUI(user_name="Tester")
    text = "## Plan\n- First step\n- Second step"

    out = ui._format_assistant_terminal_text(text)

    assert out == text
