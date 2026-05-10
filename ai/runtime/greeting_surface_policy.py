from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from ai.runtime.continuity_claim_guard import assess_continuity_claims


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
) -> GreetingSurfaceResult:
    state = dict(session_state or {})
    operator = dict(operator_state or {})
    text = str(user_input or "").strip().lower()

    greeting_count = int(state.get("greeting_count", 0) or 0)
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
    generated_by = "llm_constrained"
    style = "llm_constrained"
    reason = "llm_candidate_accepted"
    used_objective = bool(continuation_requested and has_active_focus)
    continuity_metadata: dict[str, Any] = {}
    llm_candidate_rejected = False

    llm_candidate = ""
    llm_reasons: list[str] = []
    allow_focus_reference = bool(continuation_requested and has_active_focus)
    if llm_generate:
        llm_candidate, llm_reasons, continuity_metadata = _try_constrained_llm_greeting(
            llm_generate=llm_generate,
            user_name=user_name,
            allow_focus_reference=allow_focus_reference,
            current_focus=current_focus if allow_focus_reference else "",
            repeated_greeting=repeated_greeting,
        )

    if llm_candidate:
        rendered = llm_candidate
    else:
        rendered = _llm_failure_minimal(user_name=user_name)
        generated_by = "fallback"
        style = "llm_failure_minimal"
        reason = "llm_candidate_rejected"
        llm_candidate_rejected = True
        validation_passed = False
        validation_reasons.extend(llm_reasons or ["unsafe_llm_greeting_rejected"])

    if _looks_assistant_like_task_prompt(rendered):
        rendered = _llm_failure_minimal(user_name=user_name)
        generated_by = "fallback"
        style = "llm_failure_minimal"
        reason = "assistant_like_task_prompt"
        assistant_like_prompt_suppressed = True
        validation_passed = False
        validation_reasons.append("assistant_service_language")

    now = datetime.now(timezone.utc).isoformat()
    next_state = dict(state)
    next_state["last_greeting_turn"] = int(next_state.get("last_greeting_turn", 0)) + 1
    next_state["last_greeting_text"] = rendered
    next_state["greeting_count"] = greeting_count + 1
    next_state["last_greeting_at"] = now
    next_state["recent_active_objective"] = bool(has_active_focus)

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
        continuity_claims=dict(continuity_metadata or {}),
        llm_candidate_rejected=llm_candidate_rejected,
    )


def validate_chat_greeting(text: str, *, pure_greeting: bool = True) -> GreetingValidationResult:
    normalized = str(text or "").strip()
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


def _llm_failure_minimal(*, user_name: str) -> str:
    first = _first_name(user_name) or "there"
    return f"Hey {first}."


def _try_constrained_llm_greeting(
    *,
    llm_generate: Callable[..., str],
    user_name: str,
    allow_focus_reference: bool,
    current_focus: str,
    repeated_greeting: bool,
) -> tuple[str, list[str], dict[str, Any]]:
    prompt = (
        f"Write a natural greeting from Alice to {user_name or 'Gabriel'}.\n"
        f"Repeated_greeting={repeated_greeting}.\n"
        "Use 1-3 short sentences.\n"
        "If repeated_greeting=true, keep it shorter than a first greeting.\n"
        "Sound warm, normal, and familiar.\n"
        "This is only a greeting, not a task response.\n\n"
        "Do not mention previous conversations.\n"
        "Do not say 'last time', 'we were discussing', 'you mentioned', or 'conversation history'.\n"
        "Do not mention memory.\n"
        "Do not mention machine learning or any recalled topic.\n"
        "Do not mention project status, current objective, current focus, or next best move.\n"
        "Do not say 'how can I help', 'what can I do', 'what do you need', or 'anything you need'.\n"
        "Do not use forced phrases like 'let's keep it simple', 'let's move', 'I have the thread', or 'nothing caught fire'.\n"
        "If allow_focus_reference=true, you may mention current_focus once.\n"
        "Return only the greeting.\n"
        f"allow_focus_reference={allow_focus_reference}; "
        f"current_focus={current_focus if allow_focus_reference else ''}."
    )
    try:
        candidate = str(llm_generate(prompt=prompt) or "").strip()
    except TypeError:
        try:
            candidate = str(llm_generate(prompt) or "").strip()
        except Exception:
            return ("", ["llm_error"], {})
    except Exception:
        return ("", ["llm_error"], {})

    if not candidate:
        return ("", ["empty_candidate"], {})

    continuity = assess_continuity_claims(
        text=candidate,
        memory_items=[],
        operator_state={"current_focus": current_focus} if allow_focus_reference else {},
    )
    continuity_meta = dict(continuity.metadata() or {})
    if continuity.unsupported_continuity_claim:
        return ("", ["fake_continuity"], continuity_meta)

    validation = validate_chat_greeting(candidate, pure_greeting=True)
    if not validation.valid:
        return ("", list(validation.reasons), continuity_meta)
    return (validation.cleaned_text, [], continuity_meta)


def _first_name(user_name: str) -> str:
    name = str(user_name or "").strip()
    return name.split()[0] if name else ""


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
