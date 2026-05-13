from __future__ import annotations

import re
from typing import Any, Dict


def should_answer_instead_of_clarify(
    user_input: str,
    intent: str,
    operator_state: Dict[str, Any] | None = None,
    project_memory: Dict[str, Any] | None = None,
) -> bool:
    low = str(user_input or "").lower().strip()
    normalized_intent = str(intent or "").lower().strip()
    state = dict(operator_state or {})
    project = dict(project_memory or {})
    active_learning_topic = str(
        state.get("active_learning_topic")
        or project.get("active_learning_topic")
        or ""
    ).strip()

    if not low:
        return False

    # Keep clarification for explicit missing-file operations.
    if re.search(r"\bread a file\b", low) and not re.search(r"\b[a-z0-9_./\\-]+\.[a-z0-9]{1,8}\b", low):
        return False

    risky = ("delete", "drop database", "format disk", "wipe", "bypass security")
    if any(token in low for token in risky):
        return False

    if "which " in low or "which one" in low:
        return False

    if normalized_intent.startswith("conversation:clarification_needed"):
        if any(
            token in low
            for token in (
                "beginner",
                "simple",
                "just give me something",
                "doesnt matter what",
                "doesn't matter what",
                "agentic ai companion",
                "work on alice",
                "improve alice",
                "make alice more agentic",
                "research how",
            )
        ):
            return True
        if active_learning_topic and any(
            token in low
            for token in (
                "yeah give me some information",
                "give me some information",
                "tell me more",
                "go deeper",
                "explain more",
                "give me examples",
                "continue",
                "keep going",
            )
        ):
            return True

    if any(
        token in low
        for token in (
            "work on alice",
            "ready to work on alice",
            "let's work on alice",
            "lets work on alice",
            "agentic companion",
            "beginner",
            "learn something simple",
        )
    ):
        return True

    has_objective = bool(
        str(state.get("active_objective") or "").strip()
        or str(project.get("active_objective") or "").strip()
    )
    return has_objective and normalized_intent.startswith("conversation:")
