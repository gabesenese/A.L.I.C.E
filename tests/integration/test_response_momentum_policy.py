from ai.runtime.response_momentum_policy import apply_response_momentum


def test_educational_explain_without_operator_request_stays_clean():
    out = apply_response_momentum(
        user_input="i want to learn the basics",
        response_text="Which one sounds like a good starting point to you?",
        intent="conversation:educational_explain",
        route="llm",
        operator_state={"active_objective": "Improve Alice into an agentic companion/operator"},
        project_memory={},
        local_execution={},
        next_step="",
    )
    low = out.lower()
    assert "current objective is" not in low
    assert "next best move:" not in low
    assert "which one sounds like a good starting point to you?" in low


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


def test_background_claim_is_not_rewritten_by_momentum_policy_for_casual_turn():
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
    assert "processing some interesting stuff in the background" in low


def test_operator_continue_local_turn_uses_compact_evidence_surface():
    out = apply_response_momentum(
        user_input="good, it was a cold day today so i just stayed home, now i am gonna work on alice for a bit",
        response_text=(
            "Current objective is Improve Alice into an agentic companion/operator. "
            "It's good to hear you're staying warm and cozy at home! "
            "I've taken a look at my core codebase (agent_loop.py). "
            "What would you like to tackle first."
        ),
        intent="operator:continue",
        route="local",
        operator_state={
            "active_objective": "Improve Alice into an agentic companion/operator",
            "last_recommended_action": {
                "action": "inspect_file",
                "target": "ai/runtime/operator_state.py",
                "reason": "It stores active objective, current focus, inspected files, and recommendations.",
            },
        },
        project_memory={},
        local_execution={
            "action": "code:analyze_file",
            "success": True,
            "inspected_file": "ai/runtime/agent_loop.py",
            "analysis": {"lines": 326, "classes": 5, "functions": 5},
        },
        next_step="inspect file ai/runtime/operator_state.py because It stores active objective, current focus, inspected files, and recommendations.",
    )
    low = out.lower()
    assert "warm and cozy" not in low
    assert "what would you like to tackle first" not in low
    assert "i inspected ai/runtime/agent_loop.py." in low
    assert out.count("ai/runtime/agent_loop.py") == 1
    assert "next best move: inspect ai/runtime/operator_state.py" in low
    assert "\n\n" in out


def test_code_analyze_file_local_turn_uses_operator_surface():
    out = apply_response_momentum(
        user_input="inspect operator_state.py",
        response_text="Long generic paragraph that should be compacted.",
        intent="code:analyze_file",
        route="local",
        operator_state={
            "last_recommended_action": {
                "action": "inspect_file",
                "target": "ai/memory/project_memory.py",
                "reason": "state persistence follow-up",
            }
        },
        project_memory={},
        local_execution={
            "action": "code:analyze_file",
            "success": True,
            "inspected_file": "ai/runtime/operator_state.py",
            "analysis": {"responsibility": "state management"},
        },
        next_step="inspect ai/memory/project_memory.py",
    )
    assert "I inspected ai/runtime/operator_state.py." in out
    assert "Interpretation: this is state storage, not the routing brain." in out


def test_no_inspected_claim_without_local_evidence():
    out = apply_response_momentum(
        user_input="status",
        response_text="I inspected ai/runtime/operator_state.py and found issues.",
        intent="operator:status",
        route="local",
        operator_state={},
        project_memory={},
        local_execution={"success": False},
        next_step="",
    )
    assert "I inspected" not in out
