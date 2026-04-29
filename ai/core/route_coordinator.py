"""Foundation 2 route coordination helpers.

Keeps intent decision policy out of NLPProcessor orchestration flow.
"""

from dataclasses import dataclass
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from ai.core.goal_recognizer import get_goal_recognizer


@dataclass
class RouteCoordinatorConfig:
    unknown_fallback_conf_hard: float = 0.35
    unknown_fallback_conf_soft: float = 0.45
    unknown_fallback_plaus_soft: float = 0.60
    unknown_fallback_plaus_hard: float = 0.45
    route_uncertainty_threshold: float = 0.55
    clarification_intent_confidence_threshold: float = 0.45
    clarification_confidence_min: float = 0.42
    clarification_confidence_max: float = 0.62
    conversation_category_gate_threshold: float = 0.88


class RouteCoordinator:
    """Coordinates intent stabilization, gating, and final fallback policy."""

    _contextual_reaction_gratitude_terms = (
        "thanks",
        "thank you",
        "appreciate it",
        "good to know",
        "letting me know",
    )
    _contextual_reaction_state_terms = (
        "cold",
        "sick",
        "flu",
        "fever",
        "headache",
        "under the weather",
        "not feeling well",
        "tired",
        "exhausted",
        "bipolar weather",
        "weather has been",
        "got a cold",
    )
    _contextual_reaction_request_terms = (
        "can you",
        "could you",
        "please",
        "what's",
        "what is",
        "how",
        "when",
        "where",
        "show",
        "tell me",
        "check",
        "forecast",
        "temperature",
        "temp",
        "rain",
        "snow",
        "humidity",
        "wind",
        "chance",
        "should i",
    )

    def __init__(self, config: Optional[RouteCoordinatorConfig] = None) -> None:
        if config is None:
            defaults = RouteCoordinatorConfig()
            try:
                from ai.optimization.runtime_thresholds import get_thresholds

                thresholds = get_thresholds()
                config = RouteCoordinatorConfig(
                    unknown_fallback_conf_hard=float(
                        thresholds.get(
                            "unknown_fallback_conf_hard",
                            defaults.unknown_fallback_conf_hard,
                        )
                    ),
                    unknown_fallback_conf_soft=float(
                        thresholds.get(
                            "unknown_fallback_conf_soft",
                            defaults.unknown_fallback_conf_soft,
                        )
                    ),
                    unknown_fallback_plaus_soft=float(
                        thresholds.get(
                            "unknown_fallback_plaus_soft",
                            defaults.unknown_fallback_plaus_soft,
                        )
                    ),
                    unknown_fallback_plaus_hard=float(
                        thresholds.get(
                            "unknown_fallback_plaus_hard",
                            defaults.unknown_fallback_plaus_hard,
                        )
                    ),
                    route_uncertainty_threshold=float(
                        thresholds.get(
                            "route_uncertainty_threshold",
                            defaults.route_uncertainty_threshold,
                        )
                    ),
                    clarification_intent_confidence_threshold=float(
                        thresholds.get(
                            "clarification_intent_confidence_threshold",
                            defaults.clarification_intent_confidence_threshold,
                        )
                    ),
                    clarification_confidence_min=float(
                        thresholds.get(
                            "clarification_confidence_min",
                            defaults.clarification_confidence_min,
                        )
                    ),
                    clarification_confidence_max=float(
                        thresholds.get(
                            "clarification_confidence_max",
                            defaults.clarification_confidence_max,
                        )
                    ),
                    conversation_category_gate_threshold=float(
                        thresholds.get(
                            "conversation_category_gate_threshold",
                            defaults.conversation_category_gate_threshold,
                        )
                    ),
                )
            except Exception:
                config = defaults

        self.config = config
        self.goal_recognizer = get_goal_recognizer()

    @staticmethod
    def _is_direct_informational_query(text: str) -> bool:
        low = str(text or "").strip().lower()
        if not low:
            return False
        has_intro = bool(
            re.search(
                r"\b(?:i\s+(?:want|need|would\s+like)\s+to\s+know|tell\s+me)\b",
                low,
            )
        )
        has_cue = bool(
            "?" in low
            or re.search(
                r"\b(?:what|how|why|difference\s+between|compare|comparison|define|explain)\b",
                low,
            )
        )
        return bool(has_intro and has_cue)

    def apply_initial_routing_policy(
        self,
        *,
        intent: str,
        intent_confidence: float,
        route: Optional[Any],
        parsed_command: Any,
        plugin_scores: Dict[str, float],
        semantic_intent: Optional[Tuple[str, float]],
        weighted_candidates: List[Tuple[str, float]],
        build_uncertainty_prompt: Callable[
            [Any, Any, Dict[str, float]], Dict[str, Any]
        ],
        build_top_intent_candidates: Callable[..., List[Dict[str, Any]]],
        validate_intent_plausibility: Callable[..., Tuple[float, List[str]]],
        should_force_unknown_fallback: Callable[..., bool],
        normalized_text: str,
    ) -> Tuple[str, float]:
        modifiers = parsed_command.modifiers
        normalized_intent = str(intent or "").lower().strip()
        direct_info_query = self._is_direct_informational_query(normalized_text)

        if normalized_intent == "conversation:goal_statement" and direct_info_query:
            intent = "conversation:question"
            intent_confidence = max(float(intent_confidence or 0.0), 0.84)
            modifiers["goal_statement_demoted"] = "direct_informational_query"
            normalized_intent = intent

        if normalized_intent != "conversation:goal_statement":
            if direct_info_query:
                modifiers["goal_statement_suppressed"] = "direct_informational_query"
            else:
                _goal_signal = self.goal_recognizer.detect(normalized_text)
                if _goal_signal is not None:
                    intent = "conversation:goal_statement"
                    intent_confidence = max(
                        float(intent_confidence or 0.0),
                        float(_goal_signal.confidence or 0.84),
                    )
                    modifiers["goal_statement_signal"] = {
                        "goal": _goal_signal.goal,
                        "project_direction": _goal_signal.project_direction,
                        "markers": list(_goal_signal.markers or []),
                        "source": "route_coordinator",
                    }

        uncertainty = (
            build_uncertainty_prompt(route, parsed_command, plugin_scores)
            if route is not None
            else None
        )
        goal_statement_intent = (
            intent or ""
        ).lower().strip() == "conversation:goal_statement"
        if intent.startswith("vague_") and not uncertainty:
            uncertainty = {
                "needs_clarification": True,
                "question": "Can you clarify what action and target you mean?",
                "candidate_plugins": ["notes", "email", "calendar"],
                "route_confidence": intent_confidence,
                "parsed_action": parsed_command.action,
            }

        import re as _re

        if not uncertainty and _re.search(
            r"\b(do that thing|this thing|that thing|what about that|who is that)\b",
            normalized_text.lower(),
        ):
            uncertainty = {
                "needs_clarification": True,
                "question": "Could you clarify what you want me to do?",
                "candidate_plugins": ["notes", "email", "calendar"],
                "route_confidence": intent_confidence,
                "parsed_action": parsed_command.action,
            }

        if uncertainty:
            modifiers["disambiguation"] = uncertainty
            if goal_statement_intent:
                modifiers["goal_statement_preserved"] = True
            elif (
                intent_confidence
                < self.config.clarification_intent_confidence_threshold
                and not intent.startswith(("notes:", "email:", "calendar:", "system:"))
            ):
                intent = "conversation:clarification_needed"
                intent_confidence = max(intent_confidence, 0.41)

        intent_candidates = build_top_intent_candidates(
            route=route,
            weighted_candidates=weighted_candidates,
            semantic_intent=semantic_intent,
            plugin_scores=plugin_scores,
            limit=3,
        )
        plausibility_score, plausibility_issues = validate_intent_plausibility(
            normalized_text,
            intent,
            parsed_command,
            plugin_scores,
        )
        modifiers["intent_candidates"] = intent_candidates
        modifiers["intent_plausibility"] = {
            "score": plausibility_score,
            "issues": plausibility_issues,
        }

        if should_force_unknown_fallback(
            intent=intent,
            confidence=float(intent_confidence or 0.0),
            plausibility=float(plausibility_score),
            uncertainty=uncertainty,
            text=normalized_text,
        ):
            modifiers["pending_unknown_fallback"] = True
        else:
            modifiers["pending_unknown_fallback"] = False
            modifiers["unknown_intent_fallback"] = False

        if "intent_candidates" not in modifiers:
            modifiers["intent_candidates"] = intent_candidates
        if "intent_plausibility" not in modifiers:
            modifiers["intent_plausibility"] = {
                "score": plausibility_score,
                "issues": plausibility_issues,
            }

        modifiers["routing_trace"] = route.trace if route is not None else {}
        return intent, intent_confidence

    def apply_category_gate(
        self,
        *,
        intent: str,
        intent_confidence: float,
        intent_category: str,
        parsed_command: Any,
        normalized_text: str = "",
        previous_intent: str = "",
    ) -> Tuple[str, float]:
        modifiers = parsed_command.modifiers
        prior_intent = str(previous_intent or "").strip().lower()
        lower_text = str(normalized_text or "").strip().lower()

        contextual_reaction = self._is_contextual_reaction_followup(
            text=lower_text,
            previous_intent=prior_intent,
        )
        current_intent_is_tool = (":" in str(intent or "")) and not str(
            intent
        ).startswith("conversation:")
        if contextual_reaction and current_intent_is_tool:
            modifiers["tool_execution_disabled"] = True
            modifiers["contextual_reaction_gate"] = {
                "reason": "gratitude_plus_personal_state_no_new_request",
                "previous_intent": prior_intent,
                "original_intent": str(intent or ""),
                "original_confidence": float(intent_confidence or 0.0),
            }
            routing_trace = dict(modifiers.get("routing_trace") or {})
            routing_trace.update(
                {
                    "contextual_reaction_gate": True,
                    "reason": "gratitude_plus_personal_state_no_new_request",
                    "previous_intent": prior_intent,
                }
            )
            modifiers["routing_trace"] = routing_trace
            return "conversation:personal_reaction", max(
                float(intent_confidence or 0.0), 0.82
            )

        if (
            intent_category == "conversation"
            and not intent.startswith("conversation:")
            and float(intent_confidence or 0.0)
            < self.config.conversation_category_gate_threshold
        ):
            modifiers["tool_execution_disabled"] = True
            modifiers["category_gate"] = {
                "category": "conversation",
                "original_intent": intent,
                "original_confidence": float(intent_confidence or 0.0),
                "reason": "conversation_category_gate",
            }
            intent = "conversation:general"
            intent_confidence = max(
                0.72,
                min(
                    self.config.conversation_category_gate_threshold - 0.01,
                    float(intent_confidence or 0.0),
                ),
            )
        else:
            modifiers["tool_execution_disabled"] = bool(
                modifiers.get("tool_execution_disabled", False)
            )

        if intent_category == "conversation":
            modifiers["tool_execution_disabled"] = True

        return intent, intent_confidence

    def _is_contextual_reaction_followup(
        self, *, text: str, previous_intent: str
    ) -> bool:
        if not str(previous_intent or "").startswith("weather:"):
            return False

        utterance = str(text or "").strip().lower()
        if not utterance:
            return False

        has_personal_state = any(
            marker in utterance for marker in self._contextual_reaction_state_terms
        )
        has_direct_request = "?" in utterance or any(
            marker in utterance for marker in self._contextual_reaction_request_terms
        )

        return has_personal_state and not has_direct_request

    def ensure_metadata(
        self,
        *,
        parsed_command: Any,
        route: Optional[Any],
        weighted_candidates: List[Tuple[str, float]],
        semantic_intent: Optional[Tuple[str, float]],
        plugin_scores: Dict[str, float],
        build_top_intent_candidates: Callable[..., List[Dict[str, Any]]],
        validate_intent_plausibility: Callable[..., Tuple[float, List[str]]],
        normalized_text: str,
        intent: str,
    ) -> None:
        modifiers = parsed_command.modifiers
        if "intent_candidates" not in modifiers:
            modifiers["intent_candidates"] = build_top_intent_candidates(
                route=route,
                weighted_candidates=weighted_candidates,
                semantic_intent=semantic_intent,
                plugin_scores=plugin_scores,
                limit=3,
            )
        if "intent_plausibility" not in modifiers:
            score, issues = validate_intent_plausibility(
                normalized_text,
                intent,
                parsed_command,
                plugin_scores,
            )
            modifiers["intent_plausibility"] = {"score": score, "issues": issues}
        if "unknown_intent_fallback" not in modifiers:
            modifiers["unknown_intent_fallback"] = False

    def apply_final_fallback(
        self,
        *,
        intent: str,
        intent_confidence: float,
        parsed_command: Any,
        final_plausibility: float,
        strong_action_frame: bool,
        followup_locked_final: bool,
        should_force_unknown_fallback: Callable[..., bool],
        normalized_text: str = "",
    ) -> Tuple[str, float]:
        modifiers = parsed_command.modifiers
        normalized_intent = (intent or "").lower().strip()
        if normalized_intent == "conversation:goal_statement":
            modifiers["unknown_intent_fallback"] = False
            return intent, intent_confidence

        pending_unknown_fallback = bool(
            modifiers.get("pending_unknown_fallback", False)
        )
        should_fallback_final = (
            pending_unknown_fallback
            or should_force_unknown_fallback(
                intent=intent,
                confidence=float(intent_confidence or 0.0),
                plausibility=float(final_plausibility),
                uncertainty=(
                    modifiers.get("disambiguation")
                    if isinstance(modifiers.get("disambiguation"), dict)
                    else None
                ),
                text=normalized_text,
            )
        )

        if (
            (not strong_action_frame)
            and (not followup_locked_final)
            and should_fallback_final
        ):
            modifiers["unknown_intent_fallback"] = True
            modifiers["disambiguation"] = {
                "needs_clarification": True,
                "question": (
                    "I might be misreading your request. Did you want me to perform an action, "
                    "or should we continue as a discussion?"
                ),
                "candidate_plugins": [
                    c.get("intent", "").split(":", 1)[0]
                    for c in modifiers.get("intent_candidates", [])
                    if ":" in str(c.get("intent", ""))
                ][:3],
                "route_confidence": float(intent_confidence or 0.0),
                "intent_plausibility": float(final_plausibility),
                "fallback_reason": "unknown_intent_fallback_final",
            }
            intent = "conversation:clarification_needed"
            intent_confidence = max(
                self.config.clarification_confidence_min,
                min(
                    self.config.clarification_confidence_max,
                    float(intent_confidence or 0.0),
                ),
            )
        else:
            modifiers["unknown_intent_fallback"] = bool(
                modifiers.get("unknown_intent_fallback", False)
            )

        return intent, intent_confidence
