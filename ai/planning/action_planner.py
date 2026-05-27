"""Lightweight action planning and replanning helpers for UnifiedActionEngine."""

from __future__ import annotations

import re
from typing import Any, List

# Verb patterns that introduce a discrete actionable step
_STEP_VERBS = re.compile(
    r"\b(create|build|write|add|implement|set up|configure|run|test|deploy|"
    r"fix|update|remove|delete|check|review|install|connect|enable|disable|"
    r"generate|send|fetch|get|load|save|open|close|start|stop|migrate|refactor)\b",
    re.IGNORECASE,
)

# Sequence connectives that separate steps
_SEQ_CONNECTIVES = re.compile(
    r"\s*(?:,\s*(?:and\s+)?(?:then\s+)?|;\s*|"
    r"\bthen\b\s*|"
    r"\bafter(?:\s+that)?\b\s*|"
    r"\bfinally\b\s*|"
    r"\bnext\b\s*|"
    r"\bfirst\b\s*|"
    r"\bsecond(?:ly)?\b\s*|"
    r"\bthird(?:ly)?\b\s*|"
    r"\band\s+then\b\s*|"
    r"\band\s+also\b\s*)\s*",
    re.IGNORECASE,
)


class ActionPlanner:
    def decompose(self, request: Any) -> List[str]:
        if getattr(request, "plan_steps", None):
            return list(request.plan_steps)

        goal = str(getattr(request, "goal", "") or "").strip()
        if not goal:
            return []

        # Split on sequence connectives first
        raw_parts = _SEQ_CONNECTIVES.split(goal)
        parts = [p.strip() for p in raw_parts if p and p.strip() and len(p.strip()) > 3]

        if len(parts) > 1:
            # Keep only parts that contain an action verb or are long enough to be meaningful
            steps = [p for p in parts if _STEP_VERBS.search(p) or len(p.split()) >= 3]
            if len(steps) > 1:
                return steps[:8]  # cap at 8 steps

        # Fall back to simple "and/then" splitting, trying longest separators first
        lowered = goal.lower()
        for sep in (
            " and then ",
            " then ",
            " and also ",
            " and ",
            " next ",
            " finally ",
        ):
            if sep in lowered:
                split_parts = [part.strip() for part in goal.split(sep) if part.strip()]
                if len(split_parts) > 1:
                    return split_parts

        return [goal]

    def recovery_path(self, *, retry_count: int, retry_budget: int, partial_success: bool) -> str:
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
