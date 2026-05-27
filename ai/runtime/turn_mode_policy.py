from __future__ import annotations

import re
from typing import Any, Dict


def classify_turn_mode(
    user_input: str,
    intent: str,
    route: str,
    operator_state: Dict[str, Any] | None = None,
    project_memory: Dict[str, Any] | None = None,
) -> str:
    low = str(user_input or "").lower().strip()
    normalized_intent = str(intent or "").lower().strip()
    normalized_route = str(route or "").lower().strip()
    state = dict(operator_state or {})
    project = dict(project_memory or {})

    if normalized_intent.endswith("greeting") or normalized_intent == "greeting":
        return "greeting"
    if normalized_route == "clarify" or "clarification_needed" in normalized_intent:
        return "clarification"
    if normalized_intent in {
        "conversation:educational_explain",
        "operator:research_explain",
    }:
        return "educational_explain"
    if normalized_intent.startswith("operator:"):
        if normalized_intent == "operator:project_status":
            return "operator_status"
        return "operator_continue"
    if normalized_intent.startswith("code:") or (normalized_route == "local" and normalized_intent.startswith("code:")):
        return "code_work"
    if normalized_route in {"tool", "plugin"}:
        return "tool_result"

    if re.search(
        r"\b(what are we working on|where are we with alice|current objective|what is blocking us)\b",
        low,
    ):
        return "operator_status"
    if re.search(
        r"\b(continue|what's next|whats next|let's work on alice|lets work on alice|keep going|inspect the next file)\b",
        low,
    ):
        return "operator_continue"
    if re.search(r"\b(explain|teach me|i'm a beginner|im a beginner|show me simply)\b", low):
        return "educational_explain"
    if re.search(r"\b(analy[sz]e file|inspect codebase|read [a-z0-9_./\\-]+\.py)\b", low):
        return "code_work"
    if re.search(
        r"\b(how are you|what's up|whats up|how's it going|hows it going|hello|hi)\b",
        low,
    ):
        return "casual_companion"

    has_objective = bool(
        str(state.get("active_objective") or "").strip() or str(project.get("active_objective") or "").strip()
    )
    if has_objective and normalized_intent == "conversation:goal_statement":
        # Only escalate to operator_continue when the user intends to work,
        # not just when they're discussing a topic while an objective exists.
        _work_signals = (
            "let's",
            "lets",
            "work on",
            "implement",
            "build",
            "fix",
            "run",
            "next step",
            "continue",
            "keep going",
            "what's next",
            "whats next",
            "inspect",
            "commit",
            "push",
            "deploy",
            "apply",
        )
        if any(sig in low for sig in _work_signals):
            return "operator_continue"
    return "casual_companion"
