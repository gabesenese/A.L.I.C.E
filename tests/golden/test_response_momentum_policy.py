from ai.runtime.response_momentum_policy import apply_response_momentum


def test_educational_passive_followup_sentences_are_stripped():
    out = apply_response_momentum(
        user_input="i want to learn more about ai companion",
        response_text=(
            "So you want to know more about AI companions.\n\n"
            "They are systems that can keep context and help over time.\n\n"
            "If you want, I can keep tracking this thread and follow up next turn."
        ),
        intent="conversation:educational_explain",
        route="llm",
        operator_state={},
        project_memory={},
        local_execution={},
        next_step="",
    )
    low = out.lower()
    assert "they are systems that can keep context and help over time" in low
    assert "if you want" not in low
    assert "keep tracking this thread" not in low
    assert "follow up next turn" not in low


def test_educational_explain_does_not_receive_objective_or_next_step():
    out = apply_response_momentum(
        user_input="im trying to learn more about ai companion",
        response_text="An AI companion is a system that can build context over time.",
        intent="conversation:educational_explain",
        route="llm",
        operator_state={"active_objective": "Improve agentic companion operator runtime"},
        project_memory={},
        local_execution={},
        next_step="inspect file ai/runtime/next_step_policy.py because It decides what Alice should do after each safe step.",
    )
    low = out.lower()
    assert "current objective" not in low
    assert "next best move" not in low
    assert "ai/runtime" not in low
    assert out.strip() == "An AI companion is a system that can build context over time."


def test_clarification_does_not_receive_operator_scaffolding():
    out = apply_response_momentum(
        user_input="been great, it is friday after all and i am ready to work o alice",
        response_text=(
            "I misunderstood that response path. Please repeat your request in one line and I will answer directly. "
            "If you want, I can keep tracking this thread and follow up next turn."
        ),
        intent="conversation:clarification_needed",
        route="llm",
        operator_state={"active_objective": "Improve agentic companion operator runtime"},
        project_memory={},
        local_execution={},
        next_step="inspect file ai/runtime/next_step_policy.py because It decides what Alice should do after each safe step.",
    )
    low = out.lower()
    assert "current objective" not in low
    assert "next best move" not in low
    assert "ai/runtime" not in low
    assert "if you want" not in low
    assert "keep tracking this thread" not in low
    assert "follow up next turn" not in low
    assert "please repeat your request in one line" not in low


def test_operator_continue_still_gets_next_best_move():
    out = apply_response_momentum(
        user_input="continue",
        response_text="I inspected ai/runtime/agent_loop.py.",
        intent="operator:continue",
        route="local",
        operator_state={},
        project_memory={},
        local_execution={"success": True, "action": "code:analyze_file", "inspected_file": "ai/runtime/agent_loop.py"},
        next_step="inspect file ai/runtime/next_step_policy.py because It decides what Alice should do after each safe step.",
    )
    assert "Next best move:" in out


def test_concept_refinement_does_not_receive_operator_scaffolding():
    out = apply_response_momentum(
        user_input="i want it to be actually proactive",
        response_text=(
            "Actual proactivity means Alice needs triggers, not just responses. "
            "She should notice events, compare them to your goals, and suggest actions."
        ),
        intent="conversation:concept_refinement",
        route="llm",
        operator_state={"active_objective": "Improve agentic companion operator runtime"},
        project_memory={},
        local_execution={},
        next_step="inspect file ai/runtime/next_step_policy.py because It decides what Alice should do after each safe step.",
    )
    low = out.lower()
    assert "current objective" not in low
    assert "next best move" not in low
    assert "ai/runtime" not in low
    assert "triggers" in low


def test_implementation_request_can_receive_operator_bridge():
    out = apply_response_momentum(
        user_input="how do we implement this in alice?",
        response_text="We can start with one safe local inspection step.",
        intent="operator:continue",
        route="local",
        operator_state={"active_objective": "Improve agentic companion operator runtime"},
        project_memory={},
        local_execution={"success": True, "action": "code:request", "inspected_file": ""},
        next_step="inspect file ai/runtime/next_step_policy.py because It decides what Alice should do after each safe step.",
    )
    low = out.lower()
    assert "next best move:" in low
