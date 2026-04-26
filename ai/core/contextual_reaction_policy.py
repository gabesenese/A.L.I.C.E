"""Shared helpers for contextual-reaction detection and tracing."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Iterable, List, Mapping

CONTEXTUAL_REACTION_REASON = "personal_state_weather_followup_no_new_request"


@dataclass(frozen=True)
class ContextualReactionAnalysis:
    """Structured result for contextual reaction classification."""

    is_contextual_reaction: bool
    reason: str = ""
    prior_intent_weather: bool = False
    recent_weather_tool: bool = False
    weather_context: bool = False
    has_question_mark: bool = False
    direct_request_detected: bool = False
    matched_state_terms: List[str] = field(default_factory=list)
    matched_request_terms: List[str] = field(default_factory=list)
    matched_gratitude_terms: List[str] = field(default_factory=list)

    def to_trace(self) -> dict[str, Any]:
        return {
            "is_contextual_reaction": bool(self.is_contextual_reaction),
            "reason": str(self.reason or ""),
            "prior_intent_weather": bool(self.prior_intent_weather),
            "recent_weather_tool": bool(self.recent_weather_tool),
            "weather_context": bool(self.weather_context),
            "has_question_mark": bool(self.has_question_mark),
            "direct_request_detected": bool(self.direct_request_detected),
            "matched_state_terms": list(self.matched_state_terms),
            "matched_request_terms": list(self.matched_request_terms),
            "matched_gratitude_terms": list(self.matched_gratitude_terms),
        }


def _dedupe_keep_order(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for raw in values:
        token = str(raw or "").strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def _contains_term(text: str, term: str) -> bool:
    phrase = str(term or "").strip().lower()
    if not phrase:
        return False

    parts = [segment for segment in phrase.split() if segment]
    if not parts:
        return False

    pattern = r"\\b" + r"\\s+".join(re.escape(part) for part in parts) + r"\\b"
    return bool(re.search(pattern, text))


def _match_terms(text: str, terms: Iterable[str]) -> List[str]:
    return _dedupe_keep_order(term for term in terms if _contains_term(text, term))


def analyze_weather_personal_reaction(
    *,
    user_input: str,
    previous_intent: str = "",
    last_tool_result: Mapping[str, Any] | None = None,
    current_turn_number: int = 0,
    last_tool_turn_number: int = 0,
    max_recent_turn_gap: int = 3,
    gratitude_terms: Iterable[str] = (),
    state_terms: Iterable[str] = (),
    request_terms: Iterable[str] = (),
) -> ContextualReactionAnalysis:
    """Classify whether text is a weather-follow-up personal reaction."""

    text = str(user_input or "").strip().lower()
    if not text:
        return ContextualReactionAnalysis(is_contextual_reaction=False)

    matched_state_terms = _match_terms(text, state_terms)
    matched_request_terms = _match_terms(text, request_terms)
    matched_gratitude_terms = _match_terms(text, gratitude_terms)

    has_question_mark = "?" in text
    direct_request_detected = bool(has_question_mark or matched_request_terms)

    prior_intent_weather = str(previous_intent or "").strip().lower().startswith("weather:")

    tool_payload = dict(last_tool_result or {})
    tool_name = str(tool_payload.get("tool_name") or "").strip().lower()
    action = str(tool_payload.get("action") or "").strip().lower()
    success = bool(tool_payload.get("success"))

    weather_tool = bool(tool_name.startswith("weather") or action.startswith("weather:"))
    turn_is_recent = True

    current_turn = int(current_turn_number or 0)
    last_tool_turn = int(last_tool_turn_number or 0)
    gap_limit = max(1, int(max_recent_turn_gap or 1))
    if current_turn > 0 and last_tool_turn > 0:
        turn_gap = current_turn - last_tool_turn
        turn_is_recent = 0 <= turn_gap <= gap_limit

    recent_weather_tool = bool(success and weather_tool and turn_is_recent)
    weather_context = bool(prior_intent_weather or recent_weather_tool)

    is_contextual_reaction = bool(
        weather_context and matched_state_terms and not direct_request_detected
    )
    reason = CONTEXTUAL_REACTION_REASON if is_contextual_reaction else ""

    return ContextualReactionAnalysis(
        is_contextual_reaction=is_contextual_reaction,
        reason=reason,
        prior_intent_weather=prior_intent_weather,
        recent_weather_tool=recent_weather_tool,
        weather_context=weather_context,
        has_question_mark=has_question_mark,
        direct_request_detected=direct_request_detected,
        matched_state_terms=matched_state_terms,
        matched_request_terms=matched_request_terms,
        matched_gratitude_terms=matched_gratitude_terms,
    )
