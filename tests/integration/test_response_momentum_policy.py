from ai.runtime.response_momentum_policy import apply_response_momentum


def test_momentum_rewrites_passive_beginner_question_into_next_move():
    out = apply_response_momentum(
        response_text="Which one sounds like a good starting point to you?",
        intent="conversation:educational_explain",
        route="llm",
        operator_state={"active_objective": "Improve Alice into an agentic companion/operator"},
        project_memory={},
        local_execution={},
        next_step="",
    )
    low = out.lower()
    assert "which one sounds like a good starting point to you?" not in low
    assert "next best move:" in low


def test_casual_how_are_you_does_not_get_objective_or_next_step_injection():
    out = apply_response_momentum(
        user_input="how are you?",
        response_text="I'm doing great, thanks for asking.",
        intent="conversation:general",
        route="llm",
        operator_state={
            "active_objective": "Improve Alice into an agentic companion/operator",
            "current_focus": "agent_loop.py",
            "next_recommended_action": "inspect ai/runtime/agent_loop.py",
        },
        project_memory={},
        local_execution={},
        next_step="inspect ai/runtime/agent_loop.py",
    )
    low = out.lower()
    assert "current objective is" not in low
    assert "next best move" not in low
    assert "current focus" not in low


def test_background_claim_is_rewritten_for_casual_turn_without_evidence():
    out = apply_response_momentum(
        user_input="how are you?",
        response_text="Been processing some interesting stuff in the background.",
        intent="conversation:general",
        route="llm",
        operator_state={"active_objective": "Improve Alice"},
        project_memory={},
        local_execution={},
        next_step="",
    )
    low = out.lower()
    assert "processing some interesting stuff in the background" not in low
    assert "i'm good" in low
