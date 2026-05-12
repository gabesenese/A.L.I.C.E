from __future__ import annotations

import re
from typing import Any, Dict
from ai.runtime.turn_mode_policy import classify_turn_mode
from ai.runtime.operator_response_surface import render_operator_response


def _has_local_evidence_for_inspection(local: Dict[str, Any]) -> bool:
    return bool(local and local.get("success") and str(local.get("inspected_file") or "").strip())


def _enforce_claim_evidence(text: str, local: Dict[str, Any]) -> str:
    out = str(text or "")
    low = out.lower()
    if "i inspected" in low and not _has_local_evidence_for_inspection(local):
        out = re.sub(r"\bi inspected\b", "I reviewed", out, flags=re.IGNORECASE)
    if "i deleted" in low and not bool(local.get("deleted_file")):
        out = re.sub(r"\bi deleted\b", "I can delete", out, flags=re.IGNORECASE)
    if any(token in low for token in ("creator has been open", "known to have used")):
        out = re.sub(
            r"creator has been open[^.]*\.?",
            "There is no verified public implementation record for that claim.",
            out,
            flags=re.IGNORECASE,
        )
        out = re.sub(
            r"known to have used[^.]*\.?",
            "There is no canonical implementation list for that fictional system.",
            out,
            flags=re.IGNORECASE,
        )
    return out


def apply_response_momentum(
    *,
    user_input: str = "",
    response_text: str,
    intent: str,
    route: str,
    operator_state: Dict[str, Any] | None = None,
    project_memory: Dict[str, Any] | None = None,
    local_execution: Dict[str, Any] | None = None,
    next_step: str = "",
) -> str:
    text = str(response_text or "").strip()
    low = text.lower()
    if not text:
        return text

    state = dict(operator_state or {})
    project = dict(project_memory or {})
    local = dict(local_execution or {})
    objective = str(state.get("active_objective") or project.get("active_objective") or "").strip()
    focus = str(state.get("current_focus") or project.get("current_focus") or "").strip()
    normalized_intent = str(intent or "").strip().lower()
    turn_mode = classify_turn_mode(
        user_input=user_input,
        intent=normalized_intent,
        route=str(route or ""),
        operator_state=state,
        project_memory=project,
    )
    momentum_turn = turn_mode in {
        "operator_status",
        "operator_continue",
        "educational_explain",
        "code_work",
        "tool_result",
    }
    operator_turn = normalized_intent.startswith("operator:") or momentum_turn or (
        str(route or "") == "local"
        and str(intent or "").startswith("code:")
        and bool(local)
    )

    # Ban unsupported background-work claims in casual/greeting turns.
    background_claims = (
        "processing some interesting stuff in the background",
        "working behind the scenes",
        "been monitoring",
        "been checking",
        "i was analyzing",
        "i inspected",
        "i reviewed",
    )
    if turn_mode in {"casual_companion", "greeting"}:
        low_text = text.lower()
        if any(token in low_text for token in background_claims):
            text = "I'm good.\n\nStill focused."
            low = text.lower()
        # Never inject project momentum into casual/greeting.
        return text

    if not operator_turn:
        return text

    # For local continuation turns with actual local execution, prefer a compact
    # evidence-first operator surface over concatenating LLM chatter.
    if (
        str(route or "") == "local"
        and (
            normalized_intent in {"operator:continue", "operator:execute_recommended_action"}
            or normalized_intent.startswith("code:")
        )
        and local
    ):
        rendered = render_operator_response(
            user_input=user_input,
            base_text=text,
            operator_state=state,
            local_execution=local,
            next_step=str(next_step or ""),
        )
        if rendered:
            return _enforce_claim_evidence(rendered, local)

    # Avoid passive generic endings.
    passive_markers = (
        "let me know if you need anything else",
        "how can i help",
        "how may i assist",
        "sure, i can help with that",
        "which one sounds like a good starting point to you?",
        "what would you like to start with?",
        "what would you like to tackle first",
        "which one should we inspect",
        "what would you like to focus on first",
        "if that sounds interesting",
        "if you want",
        "let me know",
    )
    for marker in passive_markers:
        if marker in low:
            text = text.replace(marker, "").strip()

    result_line = ""
    if local:
        action = str(local.get("action") or "")
        inspected = str(local.get("inspected_file") or "")
        success = bool(local.get("success"))
        if inspected and success:
            result_line = f"I inspected `{inspected}` through `{action or 'local execution'}`."
        elif action and action.startswith("code:") and success:
            result_line = f"I ran one safe local step: `{action}`."

    meaning = ""
    if str(local.get("error") or ""):
        meaning = f"That surfaced a blocker: {local.get('error')}."
    elif result_line:
        meaning = "That gives us grounded evidence for the next runtime move."

    next_line = str(next_step or "").strip()
    allow_next_step = turn_mode in {
        "operator_status",
        "operator_continue",
        "educational_explain",
        "code_work",
        "tool_result",
    }
    if next_line and allow_next_step and not next_line.lower().startswith("next"):
        next_line = f"Next best move: {next_line}"
    elif allow_next_step and not next_line and state.get("next_recommended_action"):
        next_line = f"Next best move: {state.get('next_recommended_action')}"

    allow_objective = turn_mode in {
        "operator_status",
        "operator_continue",
        "educational_explain",
        "code_work",
        "tool_result",
    }
    if objective and allow_objective:
        lead = f"Current objective is {objective}."
        if focus:
            lead = f"{lead} Current focus: {focus}."
    else:
        lead = ""

    if momentum_turn:
        low_merged = text.lower()
        if (
            "which one" in low_merged
            or "what would you like to start with" in low_merged
            or re.search(r"\?\s*$", text)
        ):
            text = re.sub(r"\s*\?\s*$", ".", text).strip()
            if not next_line:
                if "agentic" in low_merged or "companion" in low_merged or "beginner" in low_merged:
                    next_line = (
                        "Next best move: start with memory, goals, tools, and the loop; "
                        "then implement the loop first in Alice."
                    )
                elif objective:
                    next_line = f"Next best move: take one concrete step on {focus or objective}."

    text = _enforce_claim_evidence(text, local)
    parts = [p for p in [lead, text, result_line, meaning, next_line] if str(p).strip()]
    merged = " ".join(parts).strip()
    return merged or text
