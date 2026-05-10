from __future__ import annotations

import re
from typing import Any, Dict


def _brief_context_ack(user_input: str) -> str:
    text = str(user_input or "").lower()
    if "cold day" in text or ("cold" in text and "home" in text):
        return "Cold day. Good night to work on the core."
    if "stayed home" in text or "stay home" in text:
        return "Makes sense. Good time for a focused pass."
    return ""


def _extract_next_move(next_step: str, operator_state: Dict[str, Any]) -> str:
    structured = dict(operator_state.get("last_recommended_action") or {})
    target = str(structured.get("target") or "").strip()
    reason = str(structured.get("reason") or "").strip()
    action = str(structured.get("action") or "inspect_file").strip()
    if target:
        verb = "inspect" if action in {"inspect_file", "analyze_file", "read_file"} else action.replace("_", " ")
        if reason:
            return f"Next best move: {verb} {target} because {reason}."
        return f"Next best move: {verb} {target}."
    raw = str(next_step or "").strip()
    if not raw:
        return ""
    if raw.lower().startswith("next best move:"):
        return raw.rstrip(".") + "."
    return f"Next best move: {raw.rstrip('.') }."


def render_operator_response(
    *,
    user_input: str,
    base_text: str,
    operator_state: Dict[str, Any],
    local_execution: Dict[str, Any],
    next_step: str,
) -> str:
    parts: list[str] = []
    ack = _brief_context_ack(user_input)
    if ack:
        parts.append(ack)

    success = bool(local_execution.get("success"))
    inspected = str(local_execution.get("inspected_file") or "").strip()
    if success and inspected:
        parts.append(f"I inspected {inspected}.")

    analysis = dict(local_execution.get("analysis") or {})
    if analysis:
        lines = int(analysis.get("lines") or 0)
        classes = int(analysis.get("classes") or 0)
        functions = int(analysis.get("functions") or 0)
        parts.append(
            f"Finding: structure-level read shows {lines} lines, {classes} classes, and {functions} functions."
        )
        target_hint = str(next_step or "").lower()
        if "operator_state.py" in target_hint:
            parts.append("Interpretation: next weakness is likely state handoff and recommendation persistence.")
    elif str(base_text or "").strip() and not inspected:
        cleaned = re.sub(r"\s+", " ", str(base_text)).strip()
        if cleaned:
            parts.append(cleaned)

    next_move = _extract_next_move(next_step, operator_state)
    if next_move:
        parts.append(next_move)
    return "\n\n".join(p for p in parts if str(p).strip()).strip()

