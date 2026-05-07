from __future__ import annotations

from typing import Any, Dict


def apply_response_momentum(
    *,
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
    operator_turn = str(intent or "").startswith("operator:") or (
        str(route or "") == "local"
        and str(intent or "").startswith("code:")
        and bool(local)
    )
    if not operator_turn:
        return text

    # Avoid passive generic endings.
    passive_markers = (
        "let me know if you need anything else",
        "how can i help",
        "how may i assist",
        "sure, i can help with that",
    )
    for marker in passive_markers:
        if marker in low:
            text = text.replace(marker, "").strip()

    result_line = ""
    if local:
        action = str(local.get("action") or "")
        inspected = str(local.get("inspected_file") or "")
        if inspected:
            result_line = f"I inspected `{inspected}` through `{action or 'local execution'}`."
        elif action:
            result_line = f"I ran one safe local step: `{action}`."

    meaning = ""
    if str(local.get("error") or ""):
        meaning = f"That surfaced a blocker: {local.get('error')}."
    elif result_line:
        meaning = "That gives us grounded evidence for the next runtime move."

    next_line = str(next_step or "").strip()
    if next_line and not next_line.lower().startswith("next"):
        next_line = f"Next best move: {next_line}"
    elif not next_line and state.get("next_recommended_action"):
        next_line = f"Next best move: {state.get('next_recommended_action')}"

    if objective:
        lead = f"Current objective is {objective}."
        if focus:
            lead = f"{lead} Current focus: {focus}."
    else:
        lead = ""

    parts = [p for p in [lead, text, result_line, meaning, next_line] if str(p).strip()]
    merged = " ".join(parts).strip()
    return merged or text
