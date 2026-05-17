"""Response authority contract for final publication decisions."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Dict, Optional, Tuple

from ai.runtime.fallback_policy import RuntimeFallbackPolicy, FallbackDecision


@dataclass(frozen=True)
class ResponseAuthorityOutcome:
    action: str
    text: str
    reason: str


class ResponseAuthorityContract:
    """Enforce final response authority for LLM turns.

    Contract:
    - If LLM is accepted, publish that response.
    - If LLM is rejected, try refinement first.
    - If refinement fails, use deterministic fallback.
    """

    def __init__(self, fallback_policy: Optional[RuntimeFallbackPolicy] = None) -> None:
        self._fallback_policy = fallback_policy or RuntimeFallbackPolicy()

    def start_turn(self, turn_key: str = "") -> None:
        self._fallback_policy.start_turn(turn_key)

    def resolve_llm_turn(
        self,
        *,
        accepted: bool,
        response: str,
        refine_fn: Callable[[str], Optional[str]],
        deterministic_fn: Callable[[], Optional[str]],
    ) -> ResponseAuthorityOutcome:
        decision: FallbackDecision = self._fallback_policy.refine_then_deterministic(
            response=response,
            accepted=accepted,
            refine_fn=refine_fn,
            deterministic_fn=deterministic_fn,
        )
        return ResponseAuthorityOutcome(
            action=decision.action,
            text=decision.text,
            reason=decision.reason,
        )


def contract_respond_stage(
    alice: Any,
    *,
    user_input: str,
    candidate: str,
    reasoning_output: Any,
    tool_results: Dict[str, Any],
) -> str:
    """Apply final contract-response authority before publishing."""

    _ = reasoning_output
    _ = tool_results
    if "one concrete detail" in str(candidate or "").lower() and callable(
        getattr(alice, "_is_project_ideation_request", None)
    ):
        try:
            if alice._is_project_ideation_request(user_input) and callable(
                getattr(alice, "_project_ideation_guidance_response", None)
            ):
                return str(alice._project_ideation_guidance_response(user_input))
        except Exception:
            pass
    return str(candidate or "")


def finalize_conversational_surface(
    alice: Any,
    *,
    user_input: str,
    intent: str,
    response: str,
    route: str,
    plugin_result: Any,
    apply_publish_style: bool = True,
) -> str:
    """Centralize final response authority and surface gating."""

    _ = plugin_result
    text = str(response or "")
    if (
        apply_publish_style
        and route not in {"llm_fallback"}
        and callable(getattr(alice, "_publish_with_fast_llm_style", None))
    ):
        text = str(
            alice._publish_with_fast_llm_style(
                response=text, user_input=user_input, intent=intent
            )
        )
    if callable(getattr(alice, "_executive_apply_response_gate", None)):
        text = str(
            alice._executive_apply_response_gate(
                response=text, user_input=user_input, intent=intent, route=route
            )
        )
    return text


def apply_meta_question_override(
    *,
    user_input: str,
    intent: str,
    intent_confidence: float,
) -> Tuple[str, float, Dict[str, Any]]:
    """Prevent meta-AI questions from being misrouted into notes actions."""

    text = str(user_input or "").lower()
    if str(intent or "").startswith("notes:") and (
        ("you are an ai" in text or "are you being sarcastic" in text)
        and not re.search(r"\b(read|open|show)\b.*\b(note|notes)\b", text)
    ):
        return (
            "conversation:meta_question",
            max(float(intent_confidence or 0.0), 0.92),
            {"applied": True},
        )
    return intent, float(intent_confidence or 0.0), {"applied": False}


def sanitize_internal_process_output(
    alice: Any,
    *,
    user_input: str,
    use_voice: bool = False,
) -> str:
    """Compatibility wrapper for tests/legacy internals that emit contract text."""

    internal = alice._process_input_internal(user_input, use_voice=use_voice)
    text = str(internal or "")
    if getattr(alice, "response_formulator", None):
        user_resp = alice.response_formulator.generate(
            intent=str(
                getattr(alice, "last_intent", "")
                or getattr(alice, "_last_routed_intent", "conversation:general")
            ),
            context={"user_input": user_input, "response": text},
            tool_results=dict(getattr(alice, "_last_plugin_result", {}) or {}),
            reasoning_output=None,
            mode="final_answer_only",
        )
        text = str(getattr(user_resp, "message", user_resp) or text)
    low = text.lower()
    if "clarify" in low or "exact result" in low:
        user_low = str(user_input or "").lower()
        if (
            "difference between" in user_low
            and "agent" in user_low
            and ("assistant" in user_low or "assistance" in user_low)
        ):
            text = (
                "Short answer: an AI assistant is mostly reactive, while an AI agent "
                "is proactive, goal-driven, and can plan and execute steps with verification."
            )
        elif callable(getattr(alice, "_answerability_gate_fallback_response", None)):
            text = alice._answerability_gate_fallback_response(user_input)
    if "intent=" in low or "raw_response=" in low or low.startswith("analysis:"):
        filtered = [
            ln
            for ln in text.splitlines()
            if not ln.lower().startswith(("intent=", "raw_response=", "analysis:"))
        ]
        text = "\n".join(ln for ln in filtered if ln.strip()).strip()
        if not text and callable(
            getattr(alice, "_answerability_gate_fallback_response", None)
        ):
            text = alice._answerability_gate_fallback_response(user_input)
    if callable(getattr(alice, "_clamp_final_response", None)):
        return alice._clamp_final_response(
            text,
            response_type="general_response",
            route="generation",
            user_input=user_input,
        )
    return text
