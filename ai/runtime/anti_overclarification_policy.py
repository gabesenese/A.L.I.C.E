from __future__ import annotations

import re
from typing import Any, Dict

_QUESTION_START = re.compile(
    r"(how|why|what|when|where|which|who|can you|could you|should i|is it|are there|do i|does)\b"
)


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

    if not low:
        return False

    # Keep clarification for explicit missing-file operations.
    if re.search(r"\bread a file\b", low) and not re.search(r"\b[a-z0-9_./\\-]+\.[a-z0-9]{1,8}\b", low):
        return False

    # This decides answer-or-ask, not run-or-refuse; the trust tiers guard execution.
    # "Delete the logs" is an instruction worth confirming; "how do I delete a git
    # branch?" and "which file should I start with?" are questions to answer.
    is_question = low.endswith("?") or bool(_QUESTION_START.match(low))
    risky = ("delete", "drop database", "format disk", "wipe", "bypass security")
    if not is_question and any(token in low for token in risky):
        return False

    if low in {"this is unclear", "unclear"} or re.fullmatch(
        r"(this|that|it)\s+is\s+(unclear|ambiguous|confusing)", low
    ):
        return False

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

    # Conversational intents are never worth clarifying — just answer.
    # "What do you think?", "tell me more", "why?" etc. should always go to LLM.
    if normalized_intent.startswith("conversation:"):
        return True

    has_objective = bool(
        str(state.get("active_objective") or "").strip() or str(project.get("active_objective") or "").strip()
    )
    # For short social inputs (≤4 tokens) during an active session, prefer answering.
    if has_objective and len(low.split()) <= 4:
        return True

    return False
