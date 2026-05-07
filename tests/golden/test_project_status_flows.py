from __future__ import annotations

from ai.memory.project_memory import ProjectMemoryState, save_project_state
from ai.runtime.local_actions.local_action_executor import LocalActionExecutor


class _AliceStub:
    self_reflection = None
    PROJECT_ROOT = "."


def test_project_status_flow_returns_grounded_summary():
    user_id = "golden_project_status"
    save_project_state(
        ProjectMemoryState(
            active_objective="Improve agentic companion operator runtime",
            current_focus="routing",
            last_failure="routing: clarify overused",
            next_recommended_action="Inspect ai/core/routing/route_arbiter.py",
            files_inspected=["ai/core/routing/route_arbiter.py"],
        ),
        user_id=user_id,
    )
    executor = LocalActionExecutor(_AliceStub())
    result = executor.execute(
        action="operator:project_status",
        query="what are we fixing?",
        context={"user_id": user_id, "operator_state": {}},
    )
    text = str(result.get("response") or "").lower()
    assert "improving" in text
    assert "routing" in text
    assert "next move" in text
