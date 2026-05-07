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
    session_state: dict[str, Any]


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

    first_greeting = greeting_count == 0
    returning_session = bool(state.get("returning_session"))
    idle_gap_return = bool(state.get("meaningful_idle_gap"))
    allow_focus_reference = bool(
        has_active_focus
        and (continuation_requested or returning_session or idle_gap_return)
    )

    if repeated_greeting:
        rendered = _repeat_greeting(text)
        style = "repeated_greeting"
        reason = "repeated_greeting"
        used_objective = False
    elif continuation_requested and has_active_focus:
        rendered = _objective_greeting(
            user_name=user_name, user_input=text, current_focus=current_focus
        )
        style = "continuation_greeting"
        reason = "explicit_continuation_request"
        used_objective = True
    elif allow_focus_reference:
        rendered = _warm_greeting(user_name=user_name, user_input=text)
        style = "pure_greeting_with_active_state"
        reason = "active_state_present_without_forcing_continuation"
        used_objective = False
    else:
        rendered = _warm_greeting(user_name=user_name, user_input=text)
        style = "pure_greeting_no_state"
        reason = "pure_greeting_no_state"
        used_objective = False

    generated_by = "policy"
    llm_rejected = False
    if llm_generate and not repeated_greeting:
        llm_candidate = _try_constrained_llm_greeting(
            llm_generate=llm_generate,
            user_name=user_name,
            repeated_greeting=repeated_greeting,
            allow_focus_reference=allow_focus_reference,
            current_focus=current_focus if allow_focus_reference else "",
            style=style,
        )
        if llm_candidate:
            rendered = llm_candidate
            generated_by = "llm_constrained"
        else:
            llm_rejected = True

    if not _is_safe_warm_greeting(rendered):
        rendered = _warm_fallback(user_name=user_name, repeated=repeated_greeting)
        generated_by = "fallback"
        used_objective = False
        style = "fallback_warm_safe"
        reason = "unsafe_or_unusable_greeting_replaced"
    elif llm_rejected:
        generated_by = "fallback"
        reason = "unsafe_llm_greeting_rejected"

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
        session_state=next_state,
    )


def _is_pure_greeting(text: str) -> bool:
    if not text:
        return False
    normalized = " ".join(text.replace(",", " ").split())
    return normalized in {"hi", "hi alice", "hey", "hello", "hey alice", "hello alice"}


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


def _warm_greeting(*, user_name: str, user_input: str) -> str:
    first = _first_name(user_name)
    salutation = _salutation_from_input(user_input)
    if first:
        if salutation == "Hello":
            return f"Hello {first}, good to see you. What's on your mind?"
        if salutation == "Hi":
            return f"Hi {first}, good to see you. What's on your mind?"
        return f"Hey {first}, good to see you. What's on your mind?"
    return "Hey, I'm here. What are we thinking about?"


def _objective_greeting(*, user_name: str, user_input: str, current_focus: str) -> str:
    first = _first_name(user_name)
    salutation = _salutation_from_input(user_input)
    lead = f"{salutation} {first}." if first else f"{salutation}."
    return f"{lead} We were focused on Alice's {current_focus} work."


def _repeat_greeting(user_input: str) -> str:
    salutation = _salutation_from_input(user_input)
    if salutation == "Hello":
        return "Still here."
    if salutation == "Hi":
        return "Yeah, I'm here."
    return "Hey."


def _warm_fallback(*, user_name: str, repeated: bool) -> str:
    if repeated:
        return "Still here."
    first = _first_name(user_name)
    if first:
        return f"Hey {first}, good to see you."
    return "Hey, good to see you."


def _try_constrained_llm_greeting(
    *,
    llm_generate: Callable[..., str],
    user_name: str,
    repeated_greeting: bool,
    allow_focus_reference: bool,
    current_focus: str,
    style: str,
) -> str:
    prompt = (
        "Write one short warm greeting (1-2 sentences, 6-18 words preferred). "
        "Do not claim prior conversations or memory. "
        "Do not use phrases like 'last time', 'we were discussing', 'you mentioned'. "
        "Do not mention project continuation unless explicit continuation context is allowed. "
        f"user_name={user_name or 'User'}; "
        f"repeated_greeting={repeated_greeting}; "
        f"style={style}; "
        f"allow_focus_reference={allow_focus_reference}; "
        f"current_focus={current_focus if allow_focus_reference else ''}."
    )
    try:
        candidate = str(llm_generate(prompt=prompt) or "").strip()
    except TypeError:
        try:
            candidate = str(llm_generate(prompt) or "").strip()
        except Exception:
            return ""
    except Exception:
        return ""

    if not candidate:
        return ""

    continuity = assess_continuity_claims(
        text=candidate,
        memory_items=[],
        operator_state={"current_focus": current_focus} if allow_focus_reference else {},
    )
    if continuity.unsupported_continuity_claim:
        return ""
    return candidate


def _is_safe_warm_greeting(text: str) -> bool:
    normalized = str(text or "").strip()
    if not normalized:
        return False
    if normalized.count(".") + normalized.count("?") + normalized.count("!") > 3:
        return False

    low = normalized.lower()
    banned = (
        "alice's development",
        "start fresh",
        "no active task is loaded",
        "operator state",
        "memory policy",
        "broad memory",
        "how may i assist you today",
        "i am ready to assist",
        "last time we talked",
        "we were discussing",
        "conversation history suggests",
        "machine learning",
    )
    if any(token in low for token in banned):
        return False

    words = len(normalized.split())
    return 2 <= words <= 24


def _first_name(user_name: str) -> str:
    name = str(user_name or "").strip()
    return name.split()[0] if name else ""


def _salutation_from_input(text: str) -> str:
    low = str(text or "").lower()
    if "hello" in low:
        return "Hello"
    if "hi" in low:
        return "Hi"
    return "Hey"
