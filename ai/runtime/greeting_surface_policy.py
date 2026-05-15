from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable
import re

from ai.runtime.continuity_claim_guard import assess_continuity_claims
from ai.runtime.learned_response_examples import (
    LearnedResponseExample,
    find_similar_response_examples,
    record_learned_response_example,
)

# GREETING POLICY FREEZE:
# This is a locked product surface. Do not change greeting behavior without
# adding/adjusting a regression test that captures the desired user-facing behavior.


@dataclass
class GreetingSurfaceResult:
    text: str
    active_objective_used: bool
    greeting_style: str
    reason: str
    suppressed_project_menu: bool
    repeated_greeting: bool
    generated_by: str
    warmth_level: str
    companion_tone: bool
    assistant_like_prompt_suppressed: bool
    validation_passed: bool
    validation_reasons: list[str]
    session_state: dict[str, Any]
    continuity_guard_applied: bool
    continuity_claims: dict[str, Any]
    llm_candidate_rejected: bool


@dataclass
class GreetingValidationResult:
    valid: bool
    reasons: list[str]
    cleaned_text: str = ""


def render_grounded_greeting(
    *,
    user_name: str = "",
    operator_state: dict | None = None,
    session_state: dict | None = None,
    user_input: str = "",
    llm_generate: Callable[..., str] | None = None,
    local_time: datetime | None = None,
    timezone_name: str = "",
) -> GreetingSurfaceResult:
    state = dict(session_state or {})
    operator = dict(operator_state or {})
    text = str(user_input or "").strip().lower()

    greeting_count = int(state.get("greeting_count", 0) or 0)
    recent_greeting_texts = list(state.get("recent_greeting_texts") or [])
    last_greeting_text = str(state.get("last_greeting_text") or "")
    repeated_greeting = greeting_count > 0 and _is_pure_greeting(text)
    continuation_requested = _has_continuation_cue(text)
    current_focus = str(operator.get("current_focus") or "").strip()
    active_objective = str(operator.get("active_objective") or "").strip()
    has_active_focus = bool(current_focus and active_objective)

    assistant_like_prompt_suppressed = False
    warmth_level = "casual" if repeated_greeting else "warm"
    companion_tone = True
    validation_passed = True
    validation_reasons: list[str] = []
    generated_by = "none"
    style = "llm"
    reason = "llm_candidates_rejected"
    used_objective = bool(continuation_requested and has_active_focus)
    continuity_metadata: dict[str, Any] = {}
    llm_candidate_rejected = False

    llm_candidate = ""
    llm_reasons: list[str] = ["llm_unavailable"]
    llm_attempt = -1
    llm_attempt_count = 0
    allow_focus_reference = bool(continuation_requested and has_active_focus)
    resolved_local_time = local_time
    if resolved_local_time is None and str(timezone_name or "").strip():
        try:
            resolved_local_time = datetime.now().astimezone()
        except Exception:
            resolved_local_time = None
    time_period = _get_time_period(resolved_local_time)
    local_time_label = (
        resolved_local_time.strftime("%Y-%m-%d %H:%M")
        if resolved_local_time is not None
        else ""
    )
    if llm_generate:
        (
            llm_candidate,
            llm_reasons,
            continuity_metadata,
            llm_attempt,
            llm_attempt_count,
        ) = _try_constrained_llm_greeting(
            llm_generate=llm_generate,
            user_name=user_name,
            allow_focus_reference=allow_focus_reference,
            current_focus=current_focus if allow_focus_reference else "",
            repeated_greeting=repeated_greeting,
            greeting_count=greeting_count,
            last_greeting_text=last_greeting_text,
            recent_greeting_texts=recent_greeting_texts,
            time_period=time_period,
            local_time_label=local_time_label,
            user_input=user_input,
        )

    if llm_candidate:
        rendered = llm_candidate
        generated_by = "llm" if llm_attempt == 0 else "llm_retry"
        style = generated_by
        reason = (
            "llm_candidate_accepted"
            if llm_attempt == 0
            else "llm_retry_candidate_accepted"
        )
        _record_validated_greeting(
            greeting_text=rendered,
            repeated_greeting=repeated_greeting,
            user_input=user_input,
        )
    else:
        rendered = ""
        generated_by = "none"
        style = "llm_rejected"
        reason = "llm_candidates_rejected"
        llm_candidate_rejected = True
        validation_passed = False
        validation_reasons.extend(llm_reasons or ["llm_candidates_rejected"])
    assistant_like_prompt_suppressed = "assistant_service_language" in validation_reasons

    now = datetime.now(timezone.utc).isoformat()
    next_state = dict(state)
    next_state["last_greeting_turn"] = int(next_state.get("last_greeting_turn", 0)) + 1
    next_state["last_greeting_text"] = rendered
    next_state["greeting_count"] = greeting_count + 1
    next_state["last_greeting_at"] = now
    next_state["recent_active_objective"] = bool(has_active_focus)
    next_state["greeting_policy_version"] = "greeting_v1_final"
    normalized_rendered = _normalize_greeting_text(rendered)
    recent_store = [
        str(item).strip()
        for item in recent_greeting_texts
        if str(item or "").strip()
    ]
    if normalized_rendered:
        recent_store.append(normalized_rendered)
    next_state["recent_greeting_texts"] = recent_store[-5:]

    return GreetingSurfaceResult(
        text=rendered,
        active_objective_used=used_objective,
        greeting_style=style,
        reason=reason,
        suppressed_project_menu=True,
        repeated_greeting=repeated_greeting,
        generated_by=generated_by,
        warmth_level=warmth_level,
        companion_tone=companion_tone,
        assistant_like_prompt_suppressed=assistant_like_prompt_suppressed,
        validation_passed=validation_passed,
        validation_reasons=validation_reasons,
        session_state=next_state,
        continuity_guard_applied=True,
        continuity_claims={
            **dict(continuity_metadata or {}),
            "time_period": time_period or "unknown",
            "anti_repetition_checked": True,
            "greeting_memory_policy": "active_state_only",
            "broad_memory_suppressed": True,
            "greeting_policy_version": "greeting_v1_final",
            "llm_attempt_count": int(llm_attempt_count or 0),
        },
        llm_candidate_rejected=llm_candidate_rejected,
    )


def validate_chat_greeting(text: str, *, pure_greeting: bool = True) -> GreetingValidationResult:
    normalized = str(text or "").strip()
    normalized = normalized.replace("â€™", "'").replace("’", "'")
    normalized = re.sub(r"([A-Za-z])\?{2,}([A-Za-z])", r"\1'\2", normalized)
    if not normalized:
        return GreetingValidationResult(False, ["empty_greeting"], "")

    low = normalized.lower()
    sentence_count = normalized.count(".") + normalized.count("?") + normalized.count("!")
    if sentence_count < 1:
        return GreetingValidationResult(False, ["missing_sentence"], "")
    if sentence_count > 3:
        return GreetingValidationResult(False, ["too_many_sentences"], "")

    banned_tokens = (
        "how can i help",
        "what should we work on",
        "what are we working on",
        "what do you want to work on",
        "what should i help with",
        "what can i do",
        "what do you need",
        "assist",
        "support",
        "anything you need",
        "current objective",
        "current focus",
        "next best move",
        "agent_loop.py",
        "operator state",
        "back online",
        "you are online",
        "you're online",
        "last time",
        "we were discussing",
        "you mentioned",
        "conversation history",
        "we left off",
        "i remember",
        "machine learning",
        "let's keep it simple",
        "lets keep it simple",
        "let's move",
        "lets move",
        "i have the thread",
        "nothing caught fire",
        "kept the signal",
        "long time no chat",
        "long time no see",
        "been a while",
        "welcome back",
        "good to see you again",
        "nice to reconnect",
        "nice to connect with you",
    )
    if any(token in low for token in banned_tokens):
        return GreetingValidationResult(False, ["banned_content"], "")

    continuity = assess_continuity_claims(text=normalized, memory_items=[], operator_state={})
    if continuity.unsupported_continuity_claim:
        return GreetingValidationResult(False, ["fake_continuity"], "")

    words = len(normalized.split())
    if words < 1 or words > 45:
        return GreetingValidationResult(False, ["greeting_length_out_of_bounds"], "")

    if pure_greeting:
        task_intake_tokens = (
            "how can i help",
            "what can i do",
            "what do you need",
            "anything you need",
        )
        if any(token in low for token in task_intake_tokens):
            return GreetingValidationResult(False, ["assistant_service_language"], "")

    return GreetingValidationResult(True, [], normalized)


def _is_pure_greeting(text: str) -> bool:
    if not text:
        return False
    normalized = " ".join(text.replace(",", " ").split())
    return normalized in {
        "hi",
        "hi alice",
        "hey",
        "hello",
        "hey alice",
        "hello alice",
        "yo",
        "yo alice",
    }


def _has_continuation_cue(text: str) -> bool:
    cues = (
        "continue",
        "pick up",
        "where were we",
        "what's next",
        "whats next",
        "pick up where we left off",
        "let's keep going",
        "lets keep going",
    )
    return any(cue in text for cue in cues)


def _try_constrained_llm_greeting(
    *,
    llm_generate: Callable[..., str],
    user_name: str,
    allow_focus_reference: bool,
    current_focus: str,
    repeated_greeting: bool,
    greeting_count: int,
    last_greeting_text: str,
    recent_greeting_texts: list[str],
    time_period: str = "",
    local_time_label: str = "",
    user_input: str = "",
) -> tuple[str, list[str], dict[str, Any], int]:
    if not llm_generate:
        return ("", ["llm_unavailable"], {"llm_attempt_count": 0}, -1, 0)
    recent_items = [str(item).strip() for item in recent_greeting_texts if str(item or "").strip()]
    if last_greeting_text.strip():
        recent_items.append(last_greeting_text.strip())
    recent_items = recent_items[-5:]
    recent_block = "\n".join(f"- {item}" for item in recent_items) if recent_items else "- (none)"

    style_examples = _greeting_style_examples_block(
        repeated_greeting=repeated_greeting,
        user_input=user_input,
    )
    time_line = (
        f"Current time period: {time_period}. Local time: {local_time_label or 'unknown'}."
        if time_period
        else "Current time period is unknown."
    )

    def _build_initial_prompt() -> str:
        time_line_extra = ""
        if time_period:
            time_line_extra = (
                "You may lightly reflect the time of day only if it matches the provided time period.\n"
            )
        return (
            f"You are Alice speaking to {user_name or 'Gabriel'}.\n\n"
            "He just greeted you.\n"
            "Write a natural conversational greeting in your own words.\n"
            "Use only the current user message and current time if provided.\n"
            "Do not mention previous conversations, memory, history, or things he talked about before.\n"
            "Do not mention projects, files, code, tools, objectives, or next steps for a plain greeting.\n"
            "Do not ask how you can help.\n"
            "Do not sound like customer service.\n"
            "Do not be overly formal.\n"
            "Do not use task-intake language.\n"
            "Do not mention memory.\n"
            "Use 2-3 short sentences.\n"
            "It should feel like a familiar AI companion, not a support bot.\n"
            "Return only the greeting.\n\n"
            f"{time_line_extra}"
            f"{time_line}\n"
            f"allow_focus_reference={allow_focus_reference}; current_focus={current_focus if allow_focus_reference else ''}\n"
            f"repeated_greeting={repeated_greeting}; greeting_count={greeting_count}\n\n"
            "Current user message:\n"
            f"{user_input}\n\n"
            "Recent greetings to avoid:\n"
            f"{recent_block}\n\n"
            "Accepted style references (do not copy):\n"
            f"{style_examples}\n"
        )

    def _build_retry_prompt(retry_reasons: list[str]) -> str:
        reasons_line = ", ".join(sorted(set(retry_reasons or ["invalid_output"])))
        return (
            f"You are Alice speaking to {user_name or 'Gabriel'}.\n\n"
            "The previous greeting was rejected for:\n"
            f"{reasons_line}\n\n"
            "Write a new natural greeting.\n"
            "Use only the current user message and current time if provided.\n"
            "Do not mention previous conversations unless explicitly grounded.\n"
            "Do not mention projects, files, code, tools, objectives, or next steps for a plain greeting.\n"
            "Do not ask how you can help.\n"
            "Do not sound like customer service.\n"
            "Do not use task-intake language.\n"
            "Do not mention memory.\n"
            "Keep it 2-3 short sentences.\n"
            "Return only the greeting.\n\n"
            f"{time_line}\n"
            f"allow_focus_reference={allow_focus_reference}; current_focus={current_focus if allow_focus_reference else ''}\n\n"
            "Current user message:\n"
            f"{user_input}\n\n"
            "Recent greetings to avoid:\n"
            f"{recent_block}\n\n"
            "Accepted style references (do not copy):\n"
            f"{style_examples}\n"
        )

    def _build_strong_retry_prompt(retry_reasons: list[str]) -> str:
        reasons_line = ", ".join(sorted(set(retry_reasons or ["invalid_output"])))
        return (
            f"You are Alice speaking to {user_name or 'Gabriel'}.\n\n"
            "The previous greeting was rejected for:\n"
            f"{reasons_line}\n\n"
            "Your previous greeting mentioned a past topic or memory that is not grounded.\n"
            "Do not mention previous conversations, memory, history, or last time.\n"
            "Generate a fresh greeting based only on the current message.\n"
            "Avoid unsupported continuity.\n"
            "Stay warm and conversational.\n"
            "Do not mention projects, files, code, tools, objectives, or next steps.\n"
            "Do not ask how you can help.\n"
            "Do not sound like customer service.\n"
            "Use 2-3 short sentences.\n"
            "Return only the greeting.\n\n"
            f"{time_line}\n"
            "Current user message:\n"
            f"{user_input}\n\n"
            "Recent greetings to avoid:\n"
            f"{recent_block}\n\n"
            "Accepted style references (do not copy):\n"
            f"{style_examples}\n"
        )

    def _generate(prompt: str) -> str:
        try:
            return str(llm_generate(prompt=prompt) or "").strip()
        except TypeError:
            return str(llm_generate(prompt) or "").strip()

    last_reasons: list[str] = []
    continuity_meta: dict[str, Any] = {}
    max_attempts = 3
    for attempt in range(max_attempts):
        if attempt == 0:
            prompt = _build_initial_prompt()
        elif attempt == 1:
            prompt = _build_retry_prompt(last_reasons)
        else:
            prompt = _build_strong_retry_prompt(last_reasons)
        try:
            candidate = _generate(prompt)
        except Exception:
            continuity_meta["llm_attempt_count"] = attempt + 1
            return ("", ["llm_error"], continuity_meta, -1, attempt + 1)
        if not candidate:
            last_reasons = ["empty_candidate"]
            continue
        if _is_repeated_greeting(candidate, recent_items):
            last_reasons = ["repeated_candidate"]
            continue
        continuity = assess_continuity_claims(
            text=candidate,
            memory_items=[],
            operator_state={"current_focus": current_focus} if allow_focus_reference else {},
        )
        continuity_meta = dict(continuity.metadata() or {})
        continuity_meta.setdefault("broad_memory_suppressed", True)
        if continuity.unsupported_continuity_claim:
            last_reasons = ["fake_continuity_claim"]
            continue
        validation = validate_greeting_candidate(
            candidate=candidate,
            user_input=user_input,
            time_period=time_period,
            allow_focus_reference=allow_focus_reference,
            is_first_plain_greeting=(greeting_count == 0 and _is_pure_greeting(str(user_input or "").strip().lower())),
        )
        if not validation.valid:
            last_reasons = list(validation.reasons)
            continue
        if _looks_assistant_like_task_prompt(validation.cleaned_text):
            last_reasons = ["assistant_service_language"]
            continue
        continuity_meta["llm_attempt_count"] = attempt + 1
        if attempt > 0:
            continuity_meta["repetition_retry"] = True
        return (validation.cleaned_text, [], continuity_meta, attempt, attempt + 1)
    continuity_meta["llm_attempt_count"] = max_attempts
    return (
        "",
        last_reasons or ["llm_candidates_rejected"],
        continuity_meta,
        -1,
        max_attempts,
    )


def _greeting_style_examples_block(*, repeated_greeting: bool, user_input: str) -> str:
    context_signals = ["repeated_greeting"] if repeated_greeting else ["pure_greeting"]
    examples = find_similar_response_examples(
        context_signals=context_signals,
        surface="greeting",
        limit=3,
    )
    if not examples:
        return "[]"
    lines = [
        f'- "{str(ex.response_text or "").strip()}" | signals={list(ex.context_signals or [])}'
        for ex in examples
        if str(ex.response_text or "").strip()
    ]
    return "\n".join(lines) if lines else "[]"


def _record_validated_greeting(*, greeting_text: str, repeated_greeting: bool, user_input: str) -> None:
    cleaned = str(greeting_text or "").strip()
    if not cleaned:
        return
    try:
        context_signals = ["repeated_greeting"] if repeated_greeting else ["pure_greeting"]
        summary = "repeated greeting" if repeated_greeting else "plain greeting"
        record_learned_response_example(
            LearnedResponseExample.create(
                surface="greeting",
                context_signals=context_signals,
                response_text=cleaned,
                energy_signal="unknown",
                mood_signal="neutral",
                topic="",
                user_context_summary=summary,
            )
        )
    except Exception:
        return


def _normalize_greeting_text(text: str) -> str:
    raw = str(text or "").strip().lower()
    raw = raw.replace("’", "'").replace("`", "'")
    raw = raw.strip("\"' ")
    raw = re.sub(r"[^\w\s']", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


def validate_greeting_candidate(
    *,
    candidate: str,
    user_input: str,
    time_period: str,
    allow_focus_reference: bool,
    is_first_plain_greeting: bool = False,
) -> GreetingValidationResult:
    validation = validate_chat_greeting(candidate, pure_greeting=True)
    if not validation.valid:
        return validation
    low = str(candidate or "").lower()
    low = low.replace("â€™", "'").replace("’", "'")
    low = re.sub(r"([a-z])\?{2,}([a-z])", r"\1'\2", low)
    reasons: list[str] = []
    if _has_time_period_mismatch(candidate, time_period):
        reasons.append("time_period_mismatch")
    if low.count("?") > 1:
        reasons.append("too_many_questions")
    banned_map = {
        "assistant_service_language": (
            "how may i assist",
            "ready to assist",
            "support you",
        ),
        "unsupported_soft_continuity_claim": (
            "catch up",
        ),
        "corporate_language": (
            "nice to connect with you",
            "pleasure to connect",
            "touch base",
        ),
        "forced_companion_slogan": (
            "proceed",
        ),
        "project_status_in_plain_greeting": (
            "current objective",
            "next best move",
            "current focus",
            "operator state",
            "agent_loop.py",
        ),
        "device_or_online_language": (
            "back online",
            "you are online",
            "you're online",
            "system online",
        ),
    }
    for reason, tokens in banned_map.items():
        if any(token in low for token in tokens):
            reasons.append(reason)
    if "machine learning" in low and "machine learning" not in str(user_input or "").lower():
        reasons.append("stale_memory_topic")
    if not allow_focus_reference and any(
        token in low for token in ("current focus", "current objective", "routing")
    ):
        reasons.append("project_status_in_plain_greeting")

    generic_openers = (
        "great to chat",
        "nice to talk to you",
        "great to chat with you",
        "good to connect",
        "nice to meet you",
        "glad to chat",
        "great to connect",
    )
    if any(token in low for token in generic_openers):
        reasons.append("generic_empty_greeting")

    if is_first_plain_greeting:
        has_checkin = any(
            phrase in low
            for phrase in (
                "how are you",
                "how are you doing",
                "how's it going",
                "hows it going",
                "how's your",
                "hows your",
                "how's everything",
                "hows everything",
            )
        )
        has_checkin_question = has_checkin and "?" in str(candidate or "")
        has_warm_presence = any(
            phrase in low
            for phrase in (
                "good to see you",
                "good to hear from you",
                "i'm glad you're here",
                "im glad you're here",
                "i am glad you're here",
                "i'm glad you are here",
                "nice to hear from you",
                "i'm here",
                "im here",
                "i am here",
            )
        )
        words = [w for w in str(candidate or "").split() if str(w).strip()]
        word_count = len(words)
        sentence_count = str(candidate or "").count(".") + str(candidate or "").count("?") + str(candidate or "").count("!")
        normalized_low = _normalize_greeting_text(candidate)
        underspecified_exact = normalized_low in {
            "hey",
            "hi",
            "hello",
            "hey gabriel",
            "hi gabriel",
            "hello gabriel",
        }
        has_presence_signal = has_checkin_question or has_warm_presence
        has_min_structure = sentence_count >= 2 or (word_count >= 7 and has_checkin_question)
        if sentence_count > 3:
            reasons.append("too_many_sentences")
        if underspecified_exact or not has_presence_signal or not has_min_structure:
            reasons.append("underspecified_greeting")

    if reasons:
        return GreetingValidationResult(False, sorted(set(reasons)), "")
    return GreetingValidationResult(True, [], str(candidate).strip())


def _is_repeated_greeting(candidate: str, recent_greeting_texts: list[str]) -> bool:
    norm_candidate = _normalize_greeting_text(candidate)
    if not norm_candidate:
        return False
    candidate_tokens = set(norm_candidate.split())
    candidate_checkin = _extract_checkin_phrase(norm_candidate)
    candidate_opener = _extract_opening_phrase(norm_candidate)
    for recent in recent_greeting_texts:
        norm_recent = _normalize_greeting_text(recent)
        if not norm_recent:
            continue
        if norm_recent == norm_candidate:
            return True
        recent_tokens = set(norm_recent.split())
        union = candidate_tokens | recent_tokens
        overlap = (len(candidate_tokens & recent_tokens) / len(union)) if union else 0.0
        if overlap >= 0.68:
            return True
        if candidate_opener and candidate_opener == _extract_opening_phrase(norm_recent):
            return True
        if candidate_checkin and candidate_checkin == _extract_checkin_phrase(norm_recent):
            shared_openers = {"good to see you", "good to hear from you", "nice to hear from you", "how's your day going", "how are you doing"}
            if any(phrase in norm_candidate and phrase in norm_recent for phrase in shared_openers):
                return True
    return False


def _extract_checkin_phrase(normalized_text: str) -> str:
    checkins = (
        "how's your day going",
        "hows your day going",
        "how are you",
        "how are you doing",
        "how's it going",
        "hows it going",
        "how's everything",
        "hows everything",
    )
    for phrase in checkins:
        if phrase in normalized_text:
            return phrase
    return ""


def _extract_opening_phrase(normalized_text: str) -> str:
    if not normalized_text:
        return ""
    tokens = normalized_text.split()
    if not tokens:
        return ""
    work = list(tokens)
    if work and work[0] in {"hey", "hi", "hello", "yo"}:
        work = work[1:]
    if work and len(work) >= 2 and work[0] in {"gabriel", "alice", "there", "friend"}:
        work = work[1:]
    if not work:
        work = tokens
    return " ".join(work[: min(4, len(work))])


def _get_time_period(local_time: datetime | None) -> str:
    if local_time is None:
        return ""
    hour = int(local_time.hour)
    if 5 <= hour <= 11:
        return "morning"
    if 12 <= hour <= 16:
        return "afternoon"
    if 17 <= hour <= 21:
        return "evening"
    return "night"


def _has_time_period_mismatch(text: str, time_period: str) -> bool:
    low = str(text or "").lower()
    has_morning = "morning" in low
    has_afternoon = "afternoon" in low
    has_evening = "evening" in low
    has_night = bool(re.search(r"\bnight\b", low))
    has_tonight = "tonight" in low
    has_time_phrase = has_morning or has_afternoon or has_evening or has_night or has_tonight
    if not has_time_phrase:
        return False
    period = str(time_period or "").strip().lower()
    if not period:
        return True
    if has_morning and period != "morning":
        return True
    if has_afternoon and period != "afternoon":
        return True
    if has_evening and period != "evening":
        return True
    if has_night and period != "night":
        return True
    if has_tonight and period not in {"evening", "night"}:
        return True
    return False


def _looks_assistant_like_task_prompt(text: str) -> bool:
    low = str(text or "").lower()
    triggers = (
        "how can i help",
        "how may i assist",
        "what can i do for you",
        "ready to assist",
        "what's on your mind",
        "i'm here to help",
        "i am here to help",
        "here to help",
        "help with anything",
        "anything you need",
        "let me know what you need",
    )
    return any(token in low for token in triggers)
