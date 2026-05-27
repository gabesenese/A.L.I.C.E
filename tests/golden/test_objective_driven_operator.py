from __future__ import annotations

from ai.runtime.agent_loop import AgentLoop


def test_objective_driven_operator_continue_returns_next_step_or_blocker():
    loop = AgentLoop()
    result = loop.run(
        user_input="continue",
        operator_state={
            "active_objective": "Improve agentic companion operator runtime",
            "current_focus": "agent loop",
        },
        project_memory={"active_objective": "Improve agentic companion operator runtime"},
        routing_result={
            "route": "local",
            "intent": "operator:continue",
            "local_execution": {"success": True, "action": "operator:continue"},
        },
        available_files=["ai/runtime/agent_loop.py"],
        max_steps=1,
        user_id="golden_objective_driven_operator",
    ).to_dict()
    assert result["active"] is True
    assert result["next_step"] or result["blocked_reason"]
