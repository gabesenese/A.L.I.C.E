from ui.rich_terminal import RichTerminalUI


def test_long_plain_response_gets_sentence_spacing_for_terminal_readability():
    ui = RichTerminalUI(user_name="Tester")
    text = (
        "I can map a practical AI build path from here. "
        "A solid next set of tracks is intent routing, state handling, and verification strategy. "
        "Which direction should we prioritize first: architecture, tooling, or testing?"
    )

    out = ui._format_assistant_terminal_text(text)

    assert "\n\n" in out
    assert out.startswith("I can map a practical AI build path from here.")


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


def test_long_structured_single_line_response_gets_section_and_list_breaks():
    ui = RichTerminalUI(user_name="Tester")
    text = (
        "Project Ideation for AI Project Project Concept: Develop an AI-powered system that enables "
        "agentic autonomy, allowing it to take actions independently based on its understanding of the "
        "environment. Objective: The objective of this project is to create an AI project that focuses on "
        "agentic autonomy. Project Direction: Agentic Autonomy Domain: Artificial Intelligence (AI) "
        "Key Features: 1. Autonomous Decision-Making: The AI system should be able to make decisions based "
        "on its understanding of the environment and take actions accordingly. 2. Self-Improvement: The "
        "system should have the ability to learn from its experiences and improve its performance over time."
    )

    out = ui._format_assistant_terminal_text(text)

    assert "\n\nProject Concept:" in out
    assert "\n\nObjective:" in out
    assert "\n1. Autonomous Decision-Making:" in out
    assert "\n2. Self-Improvement:" in out
