from ai.runtime.response_momentum_policy import apply_response_momentum


def test_operator_status_injection_allowed_for_status_turn():
    out = apply_response_momentum(
        user_input="where are we with alice?",
        response_text="We can continue from the routing slice.",
        intent="operator:project_status",
        route="local",
        operator_state={
            "active_objective": "Improve Alice into an agentic companion/operator",
            "current_focus": "agent_loop.py",
            "next_recommended_action": "inspect ai/runtime/agent_loop.py",
        },
        project_memory={},
        local_execution={"action": "operator:project_status"},
        next_step="inspect ai/runtime/agent_loop.py",
    )
    low = out.lower()
    assert "current objective is" in low
    assert "current focus" in low
    assert "next best move" in low
