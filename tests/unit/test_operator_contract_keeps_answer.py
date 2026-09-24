"""A shape check on an operator reply does not replace it with a promise to retry."""

from ai.runtime.response_momentum_policy import apply_response_momentum


def _render(text):
    return apply_response_momentum(
        user_input="what does app/main.py do?",
        response_text=text,
        intent="operator:continue",
        route="local",
        operator_state={},
        project_memory={},
        local_execution={"action": "code:analyze_file", "success": True, "inspected_file": "app/main.py"},
    )


def test_an_answer_with_a_let_me_know_is_kept():
    out = _render("app/main.py wires the pipeline and the terminal together. Let me know if you want the routing part.")

    assert "Let me try again" not in out
    assert "wires the pipeline" in out


def test_a_short_answer_is_not_swapped_for_a_label():
    # "It runs." was under the 8-character bar and came back as "I looked at
    # app/main.py." -- or, past the renderer, as a promise to try again.
    assert _render("It runs.") == "It runs."
    assert _render("Yes.") == "Yes."


def test_a_made_up_background_claim_is_dropped_not_the_whole_reply():
    out = apply_response_momentum(
        user_input="how are you?",
        response_text="Doing well, thanks. I've been monitoring your repo all night. How was the gig?",
        intent="conversation:general",
        route="llm",
    )

    assert "monitoring" not in out
    assert "Doing well, thanks." in out and "How was the gig?" in out
