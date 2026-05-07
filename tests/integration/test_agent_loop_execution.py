from __future__ import annotations

from ai.runtime.agent_loop import AgentLoop


def test_agent_loop_executes_safe_step_for_continue():
    loop = AgentLoop()
    result = loop.run(
        user_input="continue",
        operator_state={
            "active_objective": "Improve agentic companion operator runtime",
            "current_focus": "routing",
            "last_inspected_file": "ai/core/routing/route_arbiter.py",
        },
        project_memory={"active_objective": "Improve agentic companion operator runtime"},
        routing_result={"route": "local", "intent": "operator:continue", "local_execution": {"success": True, "action": "code:analyze_file", "inspected_file": "ai/core/routing/route_arbiter.py"}},
        available_files=["ai/core/routing/route_arbiter.py"],
        max_steps=1,
        user_id="test_agent_loop_continue",
    )
    payload = result.to_dict()
    assert payload["active"] is True
    assert payload["plan_steps"]
    assert payload["executed_steps"]
    assert payload["requires_approval"] is False


def test_agent_loop_requires_approval_for_edit():
    loop = AgentLoop()
    result = loop.run(
        user_input="edit app/main.py",
        operator_state={"active_objective": "Improve agentic companion operator runtime"},
        project_memory={"active_objective": "Improve agentic companion operator runtime"},
        routing_result={"route": "local", "intent": "operator:continue", "local_execution": {}},
        available_files=["app/main.py"],
        max_steps=1,
        user_id="test_agent_loop_approval",
    )
    payload = result.to_dict()
    assert payload["requires_approval"] is True
    assert payload["blocked_reason"] == "approval_required"
