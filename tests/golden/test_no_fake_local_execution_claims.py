from ai.runtime.response_momentum_policy import apply_response_momentum


def test_no_fake_operator_continue_local_step_wording():
    out = apply_response_momentum(
        user_input="good, let's work on alice",
        response_text="Next best move: inspect file ai/runtime/agent_loop.py because Active objective exists; agent loop should drive next safe step.",
        intent="operator:continue",
        route="local",
        operator_state={
            "active_objective": "Improve Alice into an agentic companion/operator"
        },
        project_memory={},
        local_execution={
            "action": "operator:continue",
            "success": True,
            "inspected_file": "",
        },
        next_step="",
    )
    low = out.lower()
    assert "i ran one safe local step: `operator:continue`" not in low
    assert "i inspected" not in low
