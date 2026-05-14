from __future__ import annotations

import re
from typing import Any, Dict

from ai.runtime.learned_response_examples import (
    LearnedResponseExample,
    find_similar_response_examples,
    record_learned_response_example,
)

_META_ARTIFACT_PATTERNS = (
    r"\(note:\s*i['’]?ve rewritten the response[^)]*\)",
    r"note:\s*i['’]?ve rewritten[^.\n]*\.?",
    r"i['’]?ve rewritten the response to sound[^.\n]*\.?",
    r"while keeping the same facts[^.\n]*\.?",
    r"here is a rewritten version[:\s]*",
    r"rewritten:\s*",
    r"sure,\s*here['’]?s a more natural version[:\s]*",
)
_PASSIVE_OPERATOR_LINES = (
    "what would you like to start working on",
    "what would you like to work on",
    "what should we work on",
    "where should we start",
    "ready when you are",
    "how can i help",
    "let me know",
)


def strip_meta_response_artifacts(text: str) -> str:
    cleaned = str(text or "")
    for pattern in _META_ARTIFACT_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _suppress_passive_operator_chatter(text: str) -> str:
    out_lines: list[str] = []
    for line in re.split(r"\n+", str(text or "")):
        line_clean = str(line or "").strip()
        if not line_clean:
            continue
        low = line_clean.lower()
        if any(marker in low for marker in _PASSIVE_OPERATOR_LINES):
            continue
        out_lines.append(line_clean)
    return "\n".join(out_lines).strip()


def detect_context_signal(user_input: str, perception_frame: Dict[str, Any] | None = None) -> Dict[str, Any]:
    text = str(user_input or "").strip().lower()
    pf = dict(perception_frame or {})
    signals: list[str] = []
    energy_signal = str(pf.get("user_energy_signal") or "unknown").strip().lower() or "unknown"
    mood_signal = str(pf.get("user_mood_signal") or "unknown").strip().lower() or "unknown"

    if any(token in text for token in ("woke up from a nap", "just woke up", "nap")):
        signals.append("nap")
    if "long day" in text:
        signals.append("long_day")
    if "monday" in text:
        signals.append("monday")
    if "positive" in text:
        signals.append("positive")
    if any(token in text for token in ("tired", "exhausted", "drained")):
        signals.append("tired")
    if any(token in text for token in ("going to bed", "go to bed", "bed right now")):
        signals.append("bedtime")
    if any(token in text for token in ("work on alice", "work on a.l.i.c.e", "work on alice for", "work session")):
        signals.append("work_session")

    if energy_signal == "unknown" and any(s in signals for s in ("nap", "tired", "bedtime", "long_day")):
        energy_signal = "low"
    if mood_signal == "unknown":
        if "positive" in signals:
            mood_signal = "positive"
        elif "tired" in signals:
            mood_signal = "tired"
        else:
            mood_signal = "neutral"

    topic = "Alice" if ("alice" in text or "a.l.i.c.e" in text) else ""
    summary_parts: list[str] = []
    if "nap" in signals:
        summary_parts.append("user just woke up from a nap")
    elif "long_day" in signals:
        summary_parts.append("user had a long day")
    elif "bedtime" in signals:
        summary_parts.append("user is near bedtime")
    elif "tired" in signals:
        summary_parts.append("user is tired")
    if "monday" in signals:
        summary_parts.append("it is Monday")
    if "positive" in signals:
        summary_parts.append("user is staying positive")
    if "work_session" in signals and topic:
        summary_parts.append("wants a short Alice work session")
    elif "work_session" in signals:
        summary_parts.append("wants a short work session")

    return {
        "has_context": bool(signals),
        "signals": signals,
        "energy_signal": energy_signal,
        "mood_signal": mood_signal,
        "topic": topic,
        "user_context_summary": ", ".join(summary_parts).strip(),
    }


def _examples_style_block(context_signals: list[str]) -> str:
    examples = find_similar_response_examples(
        context_signals=list(context_signals or []),
        surface="operator_context_ack",
        limit=3,
    )
    if not examples:
        return "[]"
    lines = [f'- "{ex.response_text}" | signals={list(ex.context_signals or [])}' for ex in examples]
    return "\n".join(lines)


def generate_context_acknowledgement(
    context_signal: Dict[str, Any],
    *,
    user_input: str,
    llm_generate=None,
) -> str:
    if not llm_generate:
        return ""
    if not bool((context_signal or {}).get("has_context")):
        return ""
    style_examples = _examples_style_block(list(context_signal.get("signals") or []))
    prompt = (
        "Write one short acknowledgement sentence for Alice before an operator/code-work response.\n\n"
        "Use only the current user context.\n"
        "Sound natural, calm, and focused.\n"
        "Do not ask a question.\n"
        "Do not sound like therapy.\n"
        "Do not sound motivational.\n"
        "Do not sound corporate.\n"
        "Do not be cute, poetic, or dramatic.\n"
        "Do not mention memory or previous conversations.\n"
        "Do not mention files, tools, code, or internal systems.\n"
        "Do not say how can I help.\n"
        "Do not simply announce the work.\n"
        "Do not say 'let's dive in', 'let's get started', 'let's work on', or similar.\n"
        "Do not exceed 12 words.\n\n"
        f"Context signals:\n{dict(context_signal or {})}\n\n"
        f"User message:\n{str(user_input or '').strip()}\n\n"
        "Similar accepted examples for style only:\n"
        f"{style_examples}\n\n"
        "Return only the acknowledgement sentence.\n"
        "Return an empty string if no natural acknowledgement is needed."
    )
    try:
        try:
            return str(llm_generate(prompt=prompt) or "").strip()
        except TypeError:
            return str(llm_generate(prompt) or "").strip()
    except Exception:
        return ""


def validate_context_ack(text: str, context_signal: Dict[str, Any]) -> tuple[bool, list[str]]:
    candidate = str(text or "").strip()
    if not candidate:
        return (True, [])
    reasons: list[str] = []
    words = [w for w in re.split(r"\s+", candidate) if w]
    if len(words) > 12:
        reasons.append("too_long")
    low = candidate.lower()
    therapy_phrases = (
        "that must have been hard",
        "your feelings are valid",
        "i'm sorry you feel",
    )
    motivational_phrases = (
        "you've got this",
        "proud of you",
    )
    banned_phrases = (
        "how can i help",
        "what should we work on",
        "what do you need",
        "i feel",
        "i remember",
        "last time",
        "cozy",
        "agent_loop.py",
        "runtime",
        "tool",
        "file",
        "inspect",
    )
    generic_task_kickoff = (
        "let's dive",
        "let's get started",
        "let's work on",
        "let's begin",
        "let's tackle",
        "time to work on",
        "ready to work on",
    )
    if any(phrase in low for phrase in therapy_phrases):
        reasons.append("therapy_tone")
    if any(phrase in low for phrase in motivational_phrases):
        reasons.append("motivational_tone")
    if any(phrase in low for phrase in generic_task_kickoff):
        reasons.append("generic_task_kickoff")
    for phrase in banned_phrases:
        if phrase in low:
            if phrase in {"agent_loop.py", "runtime", "tool", "file", "inspect"}:
                reasons.append("internal_reference")
    if "?" in candidate:
        reasons.append("contains_question")
    energy = str((context_signal or {}).get("energy_signal") or "").lower()
    signals = {str(s).lower().strip() for s in list((context_signal or {}).get("signals") or [])}
    if "you are tired" in low and "tired" not in signals and energy != "low":
        reasons.append("ungrounded_context_ack")
    grounding_map = {
        "nap": ("nap", "woke", "awake", "fresh", "reset", "slow start", "back up"),
        "long_day": ("long day", "long one", "low-energy", "light", "steady"),
        "bedtime": ("bedtime", "late", "night", "one pass", "wind down"),
        "tired": ("tired", "low-energy", "light", "tight", "steady"),
    }
    for signal, anchors in grounding_map.items():
        if signal in signals and not any(anchor in low for anchor in anchors):
            reasons.append("ungrounded_context_ack")
            break
    reasons = list(dict.fromkeys(reasons))
    return (len(reasons) == 0, reasons)


def _generate_context_ack_with_retry(
    *,
    context_signal: Dict[str, Any],
    user_input: str,
    llm_generate=None,
) -> tuple[str, list[str]]:
    first = generate_context_acknowledgement(
        context_signal,
        user_input=user_input,
        llm_generate=llm_generate,
    )
    first_clean = str(first or "").strip()
    valid, reasons = validate_context_ack(first_clean, context_signal)
    if first_clean and valid:
        return (first_clean, [])
    if not llm_generate:
        return ("", reasons)

    style_examples = _examples_style_block(list(context_signal.get("signals") or []))
    stricter_prompt = (
        "Write one short acknowledgement sentence for Alice before an operator/code-work response.\n\n"
        "Use only the current user context.\n"
        "Sound natural, calm, and focused.\n"
        "Do not ask a question.\n"
        "Do not sound like therapy.\n"
        "Do not sound motivational.\n"
        "Do not sound corporate.\n"
        "Do not be cute, poetic, or dramatic.\n"
        "Do not mention memory or previous conversations.\n"
        "Do not mention files, tools, code, or internal systems.\n"
        "Do not say how can I help.\n"
        "Do not simply announce the work.\n"
        "Do not say 'let's dive in', 'let's get started', 'let's work on', or similar.\n"
        "Do not exceed 12 words.\n\n"
        f"Context signals:\n{dict(context_signal or {})}\n\n"
        f"User message:\n{str(user_input or '').strip()}\n\n"
        "Similar accepted examples for style only:\n"
        f"{style_examples}\n\n"
        "Your previous acknowledgement was rejected because:\n"
        f"{', '.join(reasons) if reasons else 'invalid_output'}\n\n"
        "It was too generic or not grounded in the user context.\n"
        "Write one short sentence that acknowledges the human context, not the task.\n"
        "Return an empty string if no safe natural acknowledgement is needed."
    )
    try:
        try:
            second = str(llm_generate(prompt=stricter_prompt) or "").strip()
        except TypeError:
            second = str(llm_generate(stricter_prompt) or "").strip()
    except Exception:
        return ("", reasons)
    valid_second, reasons_second = validate_context_ack(second, context_signal)
    if second and valid_second:
        return (second, [])
    return ("", reasons_second or reasons)


def _extract_next_move(next_step: str, operator_state: Dict[str, Any]) -> str:
    def _normalize_reason(raw_reason: str) -> str:
        cleaned = re.sub(r"\.+$", "", str(raw_reason or "").strip())
        if not cleaned:
            return ""
        return cleaned[0].lower() + cleaned[1:]

    def _normalize_sentence(raw: str) -> str:
        cleaned = re.sub(r"\.+$", "", str(raw or "").strip())
        if not cleaned:
            return ""
        return cleaned + "."

    structured = dict(operator_state.get("last_recommended_action") or {})
    target = str(structured.get("target") or "").strip()
    reason = str(structured.get("reason") or "").strip()
    action = str(structured.get("action") or "inspect_file").strip()
    if target:
        verb = "inspect" if action in {"inspect_file", "analyze_file", "read_file"} else action.replace("_", " ")
        if reason:
            cleaned_reason = _normalize_reason(reason)
            if cleaned_reason:
                return _normalize_sentence(f"Next best move: {verb} {target} because {cleaned_reason}")
        return _normalize_sentence(f"Next best move: {verb} {target}")
    raw = str(next_step or "").strip()
    if not raw:
        return ""
    if raw.lower().startswith("next best move:"):
        return _normalize_sentence(raw)
    return _normalize_sentence(f"Next best move: {raw}")


def render_local_execution_error_response(
    *,
    user_input: str,
    base_text: str,
    operator_state: Dict[str, Any],
    local_execution: Dict[str, Any],
    next_step: str,
) -> str:
    _ = user_input
    _ = base_text
    _ = operator_state
    parts: list[str] = ["I couldn't verify the local step."]
    error = str(local_execution.get("error") or "").strip()
    if error:
        parts.append(f"Blocker: {error}")
    next_move = _extract_next_move(next_step, dict(operator_state or {}))
    if next_move:
        parts.append(next_move)
    return "\n\n".join(parts).strip()


def render_operator_response(
    *,
    user_input: str,
    base_text: str,
    operator_state: Dict[str, Any],
    local_execution: Dict[str, Any],
    next_step: str,
    llm_generate=None,
    perception_frame: Dict[str, Any] | None = None,
) -> str:
    parts: list[str] = []
    base_text_clean = _suppress_passive_operator_chatter(strip_meta_response_artifacts(base_text))
    context_signal = detect_context_signal(user_input, dict(perception_frame or {}))
    if bool(context_signal.get("has_context")):
        cleaned_ack, _reasons = _generate_context_ack_with_retry(
            context_signal=context_signal,
            user_input=user_input,
            llm_generate=llm_generate,
        )
        if cleaned_ack:
            parts.append(cleaned_ack)
            try:
                skip_store = any(
                    token in str(user_input or "").lower()
                    for token in (
                        "delete memory",
                        "erase memory",
                        "forget this",
                        "privacy",
                        "private",
                    )
                )
                if not skip_store:
                    record_learned_response_example(
                        LearnedResponseExample.create(
                            surface="operator_context_ack",
                            context_signals=list(context_signal.get("signals") or []),
                            energy_signal=str(context_signal.get("energy_signal") or "unknown"),
                            mood_signal=str(context_signal.get("mood_signal") or "unknown"),
                            topic=str(context_signal.get("topic") or ""),
                            user_context_summary=str(context_signal.get("user_context_summary") or ""),
                            response_text=cleaned_ack,
                        )
                    )
            except Exception:
                pass

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
    elif str(base_text_clean or "").strip() and not inspected:
        cleaned = re.sub(r"\s+", " ", str(base_text_clean)).strip()
        if cleaned:
            parts.append(cleaned)

    next_move = _extract_next_move(next_step, operator_state)
    if next_move:
        parts.append(next_move)
    return "\n\n".join(p for p in parts if str(p).strip()).strip()

