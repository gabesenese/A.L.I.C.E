"""Foundation 2 route coordination helpers.

Keeps intent decision policy out of NLPProcessor orchestration flow.
"""

from dataclasses import dataclass
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

    def __init__(self, config: Optional[RouteCoordinatorConfig] = None) -> None:
        self.config = config or RouteCoordinatorConfig()
        self.goal_recognizer = get_goal_recognizer()

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

        if str(intent or "").lower().strip() != "conversation:goal_statement":
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
    ) -> Tuple[str, float]:
        modifiers = parsed_command.modifiers
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
