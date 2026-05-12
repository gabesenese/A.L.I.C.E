from __future__ import annotations

import re
from typing import Any, Dict


def _brief_context_ack(user_input: str) -> str:
    text = str(user_input or "").lower()
    if "long day" in text:
        return "Long day. We'll keep this light."
    if "monday" in text and "positive" in text:
        return "Monday test. We'll keep the pass focused."
    if "tired" in text:
        return "Tired session. We'll keep it tight."
    if "going to bed" in text or "go to bed" in text:
        return "Near bedtime. One clean pass."
    return ""


def _brief_context_ack_from_perception(perception_frame: Dict[str, Any], companion_state: Dict[str, Any]) -> str:
    pf = dict(perception_frame or {})
    cs = dict(companion_state or {})
    social = str(pf.get("social_context") or "").lower()
    mood = str(pf.get("user_mood_signal") or cs.get("mood_signal") or "").lower()
    energy = str(pf.get("user_energy_signal") or cs.get("energy_signal") or "").lower()
    time_ref = str(pf.get("time_reference") or cs.get("time_context") or "").lower()
    if "long day" in social:
        return "Long day. We'll keep this light."
    if "monday" in social and "positive" in social:
        return "Monday test. We'll keep the pass focused."
    if mood == "tired":
        return "Tired session. We'll keep it tight."
    if energy == "low" and time_ref == "night":
        return "Near bedtime. One clean pass."
    return ""


def _extract_next_move(next_step: str, operator_state: Dict[str, Any]) -> str:
    structured = dict(operator_state.get("last_recommended_action") or {})
    target = str(structured.get("target") or "").strip()
    reason = str(structured.get("reason") or "").strip()
    action = str(structured.get("action") or "inspect_file").strip()
    if target:
        verb = "inspect" if action in {"inspect_file", "analyze_file", "read_file"} else action.replace("_", " ")
        if reason:
            cleaned_reason = reason.rstrip(".")
            if cleaned_reason:
                cleaned_reason = cleaned_reason[0].lower() + cleaned_reason[1:]
            return f"Next best move: {verb} {target} because {cleaned_reason}."
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
    perception_frame: Dict[str, Any] | None = None,
    companion_state: Dict[str, Any] | None = None,
) -> str:
    parts: list[str] = []
    ack = _brief_context_ack_from_perception(
        dict(perception_frame or {}),
        dict(companion_state or {}),
    ) or _brief_context_ack(user_input)
    if ack:
        parts.append(ack)

    success = bool(local_execution.get("success"))
    inspected = str(local_execution.get("inspected_file") or "").strip()
    if success and inspected:
        parts.append(f"I inspected {inspected}.")

    analysis = dict(local_execution.get("analysis") or {})
    if analysis:
        responsibility = str(analysis.get("responsibility") or "").strip()
        if inspected.endswith("ai/runtime/operator_state.py") or inspected.endswith("ai\\runtime\\operator_state.py"):
            parts.append(
                "Finding: it stores active objective, current focus, inspected files, recommended actions, corrections, and design constraints."
            )
            parts.append("Interpretation: this is state storage, not the routing brain.")
        elif responsibility:
            if responsibility == "agent loop":
                parts.append(
                    "Finding: it owns the bounded operator loop: plan, act, observe, verify, and update state."
                )
            else:
                parts.append(f"Finding: primary responsibility is {responsibility}.")
    elif str(base_text or "").strip() and not inspected:
        cleaned = re.sub(r"\s+", " ", str(base_text)).strip()
        if cleaned:
            parts.append(cleaned)

    next_move = _extract_next_move(next_step, operator_state)
    if next_move:
        parts.append(next_move)
    return "\n\n".join(p for p in parts if str(p).strip()).strip()

