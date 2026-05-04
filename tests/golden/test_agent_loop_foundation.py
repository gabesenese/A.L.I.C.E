from ai.runtime.agent_loop import build_agent_loop_state


def test_agent_loop_local_code_step():
    payload = build_agent_loop_state(
        user_input="improve Alice's routing",
        route="local",
        intent="code:analyze_file",
        local_execution={"action": "code:analyze_file", "success": True},
        active_objective="Improve Alice into an agentic companion/operator",
    )
    assert payload["active"] is True
    assert payload["objective"]["text"]
    assert isinstance(payload["plan_steps"], list) and payload["plan_steps"]
    assert "execute_safe_step" in payload["executed_steps"]
    assert isinstance(payload["verification"], dict)
    assert isinstance(payload["next_step"], str)

