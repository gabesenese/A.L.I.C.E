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

    active_objective = str(operator.get("active_objective") or "").strip()
    current_focus = str(operator.get("current_focus") or "").strip()
    has_active_focus = bool(active_objective and current_focus)

    assistant_like_prompt_suppressed = False
    warmth_level = "warm"
    companion_tone = True
    validation_passed = True
    validation_reasons: list[str] = []
    used_objective = False
    generated_by = "policy"
    style = "minimal"
    reason = "default_minimal"

    if repeated_greeting:
        rendered = _repeat_greeting(text)
        style = "repeated_greeting"
        reason = "repeated_greeting"
        warmth_level = "casual"
    elif continuation_requested and has_active_focus:
        rendered = _objective_greeting(
            user_name=user_name, user_input=text, current_focus=current_focus
        )
        style = "continuation_context"
        reason = "explicit_continuation_request"
        used_objective = True
    else:
        rendered = _minimal_fallback(
            user_name=user_name,
            current_focus=current_focus if has_active_focus else "",
        )
        style = "minimal_fallback"
        reason = "pure_or_general_greeting_minimal_default"
        warmth_level = "presence"

    # LLM-first path for pure greeting turns.
    if llm_generate and not repeated_greeting and _is_pure_greeting(text):
        llm_candidate, llm_reasons = _try_constrained_llm_greeting(
            llm_generate=llm_generate,
            user_name=user_name,
            allow_focus_reference=continuation_requested and has_active_focus,
            current_focus=current_focus if continuation_requested and has_active_focus else "",
        )
        if llm_candidate:
            rendered = llm_candidate
            generated_by = "llm_constrained"
            style = "llm_constrained"
            reason = "llm_candidate_accepted"
            warmth_level = "warm"
        else:
            rendered = _minimal_fallback(
                user_name=user_name,
                current_focus=current_focus if has_active_focus else "",
            )
            generated_by = "fallback"
            style = "minimal_fallback"
            reason = "llm_candidate_rejected"
            warmth_level = "presence"
            validation_passed = False
            validation_reasons.extend(llm_reasons or ["unsafe_llm_greeting_rejected"])

    if _looks_assistant_like_task_prompt(rendered):
        rendered = _minimal_fallback(
            user_name=user_name,
            current_focus=current_focus if has_active_focus else "",
        )
        generated_by = "fallback"
        style = "minimal_fallback"
        reason = "assistant_like_task_prompt"
        warmth_level = "presence"
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
    )


def validate_chat_greeting(
    text: str, *, pure_greeting: bool = True
) -> GreetingValidationResult:
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
        "back online",
        "you're online",
        "you are online",
        "great to see",
        "nice to see",
        "what's on your mind",
        "how's your day",
        "hows your day",
        "how's everything",
        "hows everything",
        "last time",
        "we were discussing",
        "you mentioned",
        "conversation history",
        "let's get our head straight",
        "lets get our head straight",
        "let's think clearly",
        "lets think clearly",
        "i have the thread",
        "nothing caught fire",
        "kept the signal",
        "proceed",
    )
    if any(token in low for token in banned_tokens):
        return GreetingValidationResult(False, ["banned_content"], "")

    continuity = assess_continuity_claims(text=normalized, memory_items=[], operator_state={})
    if continuity.unsupported_continuity_claim:
        return GreetingValidationResult(False, ["fake_continuity"], "")

    if pure_greeting:
        task_intake_tokens = (
            "what are we doing",
            "what are we tackling",
            "what should we work on",
            "what are we working on",
        )
        if any(token in low for token in task_intake_tokens):
            return GreetingValidationResult(False, ["direct_task_intake"], "")

    words = len(normalized.split())
    if words < 2 or words > 40:
        return GreetingValidationResult(False, ["greeting_length_out_of_bounds"], "")

    return GreetingValidationResult(True, [], normalized)


def filter_learned_greetings(candidates: list[str], *, pure_greeting: bool = True) -> list[str]:
    accepted: list[str] = []
    for candidate in candidates:
        result = validate_chat_greeting(candidate, pure_greeting=pure_greeting)
        if result.valid and result.cleaned_text:
            accepted.append(result.cleaned_text)
    return accepted


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


def _objective_greeting(*, user_name: str, user_input: str, current_focus: str) -> str:
    first = _first_name(user_name)
    salutation = _salutation_from_input(user_input)
    lead = f"{salutation} {first}." if first else f"{salutation}."
    return f"{lead}\n\nStill on {current_focus}."


def _repeat_greeting(user_input: str) -> str:
    salutation = _salutation_from_input(user_input)
    if salutation == "Hello":
        return "Still here."
    if salutation == "Hi":
        return "Yeah, I'm here."
    return "Hey."


def _minimal_fallback(*, user_name: str, current_focus: str = "") -> str:
    first = _first_name(user_name)
    lead = f"Hey {first}." if first else "Hey."
    if current_focus:
        return f"{lead}\n\nStill on {current_focus}."
    return lead


def _try_constrained_llm_greeting(
    *,
    llm_generate: Callable[..., str],
    user_name: str,
    allow_focus_reference: bool,
    current_focus: str,
) -> tuple[str, list[str]]:
    prompt = (
        f"Write a natural greeting from Alice to {user_name or 'Gabriel'}.\n"
        "Use 2-3 short sentences or lines.\n"
        "It should feel calm, familiar, and present.\n"
        "Do not sound like a service assistant.\n"
        "Do not ask how to help.\n"
        "Do not offer help.\n"
        "Do not mention being online.\n"
        "Do not claim previous topics.\n"
        "Do not say 'last time', 'we were discussing', 'you mentioned', or 'conversation history'.\n"
        "Do not mention projects unless continuation context is explicitly allowed.\n"
        "Do not use dramatic, poetic, technical, or motivational phrases.\n"
        "Keep it simple.\n"
        f"allow_focus_reference={allow_focus_reference}; "
        f"current_focus={current_focus if allow_focus_reference else ''}."
    )
    try:
        candidate = str(llm_generate(prompt=prompt) or "").strip()
    except TypeError:
        try:
            candidate = str(llm_generate(prompt) or "").strip()
        except Exception:
            return ("", ["llm_error"])
    except Exception:
        return ("", ["llm_error"])

    if not candidate:
        return ("", ["empty_candidate"])

    continuity = assess_continuity_claims(
        text=candidate,
        memory_items=[],
        operator_state={"current_focus": current_focus} if allow_focus_reference else {},
    )
    if continuity.unsupported_continuity_claim:
        return ("", ["fake_continuity"])

    validation = validate_chat_greeting(candidate, pure_greeting=True)
    if not validation.valid:
        return ("", list(validation.reasons))
    return (validation.cleaned_text, [])


def _first_name(user_name: str) -> str:
    name = str(user_name or "").strip()
    return name.split()[0] if name else ""


def _salutation_from_input(text: str) -> str:
    low = str(text or "").lower()
    if "hello" in low:
        return "Hello"
    if "hi" in low:
        return "Hi"
    if "yo" in low:
        return "Hey"
    return "Hey"


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
