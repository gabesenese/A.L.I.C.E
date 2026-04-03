"""Lightweight action planning and replanning helpers for UnifiedActionEngine."""

from __future__ import annotations

from typing import Any, List


class ActionPlanner:
    def decompose(self, request: Any) -> List[str]:
        if getattr(request, "plan_steps", None):
            return list(request.plan_steps)

        goal = str(getattr(request, "goal", "") or "").strip()
        if not goal:
            return []

        lowered = goal.lower()
        for sep in (" and then ", " then ", " and "):
            if sep in lowered:
                parts = [part.strip() for part in goal.split(sep) if part.strip()]
                if len(parts) > 1:
                    return parts

        return [goal]

    def recovery_path(
        self, *, retry_count: int, retry_budget: int, partial_success: bool
    ) -> str:
        if partial_success:
            return "clarify_then_continue"
        if retry_count < max(0, retry_budget):
            return "retry"
        return "escalate"


_action_planner: ActionPlanner | None = None


def get_action_planner() -> ActionPlanner:
    global _action_planner
    if _action_planner is None:
        _action_planner = ActionPlanner()
    return _action_planner
