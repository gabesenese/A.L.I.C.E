from __future__ import annotations

import re
from typing import Any, Dict
from ai.runtime.turn_mode_policy import classify_turn_mode
from ai.runtime.operator_response_surface import (
    normalize_response_paragraphs,
    render_operator_response,
    strip_meta_response_artifacts,
)
from ai.runtime.verified_growth import verify_operator_surface_contract


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


def _is_operator_application_request(user_input: str) -> bool:
    low = str(user_input or "").lower()
    return any(
        token in low
        for token in (
            "work on alice",
            "work o alice",
            "apply this to alice",
            "implement this in alice",
            "build this into alice",
            "improve alice",
            "give me a codex input",
            "inspect",
            "codebase",
            "runtime",
            "file",
            "repo",
            "commit",
        )
    )


def _normalize_next_move_line(next_step: str) -> str:
    raw = str(next_step or "").strip()
    if not raw:
        return ""
    if not raw.lower().startswith("next best move:"):
        raw = f"Next best move: {raw}"
    raw = re.sub(
        r"\bbecause\s+([A-Z])",
        lambda m: f"because {str(m.group(1) or '').lower()}",
        raw,
    )
    raw = re.sub(r"\s+", " ", raw).strip()
    raw = re.sub(r"\.+$", "", raw) + "."
    return raw


def _is_explicitly_verified_operator_text(text: str) -> bool:
    low = str(text or "").lower()
    trusted_markers = (
        "i have not verified the codebase yet",
        "i can inspect local source code in this workspace",
        "inspect the local workspace",
        "i couldn't verify the local step.",
        "could not verify that result safely",
        "explicit approval",
    )
    return any(marker in low for marker in trusted_markers)


def strip_passive_followup_sentences(text: str, *, mode: str) -> str:
    mode_key = str(mode or "").strip().lower()
    if mode_key not in {"educational_explain", "clarification", "companion_chat", "concept_refinement"}:
        return str(text or "").strip()
    source = str(text or "").strip()
    if not source:
        return ""
    fragments = re.split(r"(?<=[.!?])\s+|\n+", source)
    banned_tokens = (
        "if you want",
        "let me know",
        "i can keep tracking this thread",
        "keep tracking this thread",
        "follow up next turn",
        "i'll follow up",
        "i can follow up",
        "i can keep track",
        "we can revisit",
        "ask me if",
        "would you like me to",
        "do you want me to",
        "please repeat your request in one line",
        "which one sounds like a good starting point",
    )
    kept: list[str] = []
    for frag in fragments:
        sentence = str(frag or "").strip()
        if not sentence:
            continue
        low = sentence.lower()
        matched_token = next((token for token in banned_tokens if token in low), "")
        if matched_token:
            cut_idx = low.find(matched_token)
            preserved = sentence[:cut_idx].strip(" ,;:-")
            if preserved:
                kept.append(preserved)
            continue
        kept.append(sentence)
    cleaned = " ".join(kept).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\s+([,.!?])", r"\1", cleaned)
    return cleaned


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
    llm_generate=None,
    perception_frame: Dict[str, Any] | None = None,
    companion_state: Dict[str, Any] | None = None,
    response_generation_metadata: Dict[str, Any] | None = None,
) -> str:
    text = strip_meta_response_artifacts(str(response_text or "").strip())
    low = text.lower()
    if not text:
        return text

    state = dict(operator_state or {})
    project = dict(project_memory or {})
    local = dict(local_execution or {})
    objective = str(state.get("active_objective") or project.get("active_objective") or "").strip()
    focus = str(state.get("current_focus") or project.get("current_focus") or "").strip()
    normalized_intent = str(intent or "").strip().lower()
    normalized_route = str(route or "").strip().lower()
    operator_application_request = _is_operator_application_request(user_input=user_input)
    turn_mode = classify_turn_mode(
        user_input=user_input,
        intent=normalized_intent,
        route=normalized_route,
        operator_state=state,
        project_memory=project,
    )
    momentum_turn = turn_mode in {
        "operator_status",
        "operator_continue",
        "code_work",
    }
    tool_operator_turn = (
        normalized_route in {"tool", "plugin"}
        and (
            normalized_intent.startswith("operator:")
            or normalized_intent.startswith("code:")
        )
    )
    operator_turn = momentum_turn or (
        normalized_intent.startswith("operator:")
        or normalized_intent.startswith("code:")
        or normalized_route == "local"
        or tool_operator_turn
        or (
            normalized_intent == "conversation:educational_explain"
            and operator_application_request
        )
    )

    if turn_mode == "greeting":
        # Never inject project momentum into casual/greeting.
        return text
    if turn_mode == "casual_companion":
        background_claims = (
            "processing some interesting stuff in the background",
            "working behind the scenes",
            "been monitoring",
            "been checking",
            "i was analyzing",
            "i inspected",
            "i reviewed",
        )
        if any(token in low for token in background_claims):
            return "I'm good.\n\nStill focused."
        return normalize_response_paragraphs(_enforce_claim_evidence(text, local))

    if (
        normalized_intent == "conversation:clarification_needed"
        and normalized_route == "llm"
    ):
        cleaned = strip_passive_followup_sentences(text, mode="clarification")
        if not cleaned:
            cleaned = text
        return normalize_response_paragraphs(_enforce_claim_evidence(cleaned, local))

    if (
        normalized_intent == "conversation:educational_explain"
        and not operator_application_request
    ):
        cleaned = strip_passive_followup_sentences(text, mode="educational_explain")
        if not cleaned and objective:
            cleaned = _normalize_next_move_line(f"take one concrete step on {focus or objective}")
        elif not cleaned:
            cleaned = text
        return normalize_response_paragraphs(_enforce_claim_evidence(cleaned, local))

    if (
        normalized_intent == "conversation:concept_refinement"
        and normalized_route == "llm"
        and not operator_application_request
    ):
        cleaned = strip_passive_followup_sentences(text, mode="concept_refinement")
        if not cleaned:
            cleaned = text
        return normalize_response_paragraphs(_enforce_claim_evidence(cleaned, local))

    if not operator_turn:
        return text

    operator_contract_turn = (
        normalized_route == "local"
        or normalized_intent.startswith("operator:")
    )
    if operator_contract_turn:
        rendered = render_operator_response(
            user_input=user_input,
            base_text=text,
            operator_state=state,
            local_execution=local,
            next_step=str(next_step or ""),
            llm_generate=llm_generate,
            perception_frame=dict(perception_frame or {}),
            response_metadata=response_generation_metadata,
            base_text_is_verified=_is_explicitly_verified_operator_text(text),
        )
        if normalized_intent in {"operator:project_status", "operator:status"}:
            status_parts: list[str] = []
            if objective:
                status_parts.append(f"current objective is {objective}.")
            if focus:
                status_parts.append(f"current focus: {focus}.")
            status_text = " ".join(status_parts).strip()
            if status_text:
                if "finding:" in str(rendered or "").lower():
                    rendered = f"{rendered}\n\n{status_text}"
                else:
                    rendered = f"Finding: {status_text}\n\n{rendered}".strip()
        if not str(rendered or "").strip():
            return (
                "I couldn't verify the operator response surface.\n\n"
                "Blocker: operator response contract failed.\n\n"
                "Next best move: inspect ai/runtime/operator_response_surface.py."
            )
        contract = verify_operator_surface_contract(
            route=normalized_route,
            intent=normalized_intent,
            response_text=rendered,
            local_execution=local,
            next_step=str(next_step or ""),
        )
        if contract.passed:
            return normalize_response_paragraphs(_enforce_claim_evidence(rendered, local))
        return (
            "I couldn't verify the operator response surface.\n\n"
            "Blocker: operator response contract failed.\n\n"
            "Next best move: inspect ai/runtime/operator_response_surface.py."
        )

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
        "i can keep tracking this thread",
        "follow up next turn",
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
    allow_next_step = (
        operator_application_request
        or normalized_intent.startswith("operator:")
        or normalized_intent.startswith("code:")
        or normalized_route == "local"
    ) and (
        turn_mode in {
            "operator_status",
            "operator_continue",
            "code_work",
            "tool_result",
        }
    )
    if next_line and allow_next_step:
        next_line = _normalize_next_move_line(next_line)
    elif allow_next_step and not next_line and state.get("next_recommended_action"):
        next_line = _normalize_next_move_line(str(state.get("next_recommended_action") or ""))
    else:
        next_line = ""

    allow_objective = (
        operator_application_request
        or normalized_intent.startswith("operator:")
        or normalized_intent.startswith("code:")
        or normalized_route == "local"
    ) and (
        turn_mode in {
            "operator_status",
            "operator_continue",
            "code_work",
            "tool_result",
        }
    )
    if objective and allow_objective:
        lead = f"Current objective is {objective}."
        if focus:
            lead = f"{lead} Current focus: {focus}."
    else:
        lead = ""

    if momentum_turn and allow_next_step:
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
    return normalize_response_paragraphs(merged or text)
