"""
Executive control layer for high-level action decisions.

This module keeps a structured internal reasoning state and outputs a compact
decision signal for routing behavior. It is intentionally not chain-of-thought.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ReasoningState:
    user_intent: str
    topic: str
    confidence: float
    conversation_goal: str
    user_goal: str
    depth_level: int
    plan: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "user_intent": self.user_intent,
            "topic": self.topic,
            "confidence": self.confidence,
            "conversation_goal": self.conversation_goal,
            "user_goal": self.user_goal,
            "depth_level": self.depth_level,
            "plan": list(self.plan),
        }


@dataclass
class ExecutiveDecision:
    action: str  # use_plugin | use_llm | ask_clarification | ignore | answer_direct
    reason: str
    store_memory: bool = True
    clarification_question: str = ""


class ExecutiveController:
    """Produces high-level decisions from compact state and runtime hints."""

    TOOL_DOMAINS = (
        "notes:",
        "email:",
        "calendar:",
        "file_operations:",
        "memory:",
        "reminder:",
        "system:",
        "weather:",
        "time:",
    )

    def build_state(
        self,
        user_input: str,
        intent: str,
        confidence: float,
        entities: Dict[str, Any],
        conversation_state: Dict[str, Any],
    ) -> ReasoningState:
        entities = entities or {}
        conversation_state = conversation_state or {}

        topic = str(
            entities.get("topic")
            or conversation_state.get("conversation_topic")
            or ""
        ).strip()

        conversation_goal = str(conversation_state.get("conversation_goal") or "").strip()
        user_goal = str(conversation_state.get("user_goal") or entities.get("goal") or "").strip()
        depth_level = int(conversation_state.get("depth_level") or 0)

        plan = self._derive_plan(intent=intent, topic=topic, depth_level=depth_level, user_input=user_input)

        return ReasoningState(
            user_intent=intent or "unknown",
            topic=topic,
            confidence=max(0.0, min(1.0, float(confidence or 0.0))),
            conversation_goal=conversation_goal,
            user_goal=user_goal,
            depth_level=depth_level,
            plan=plan,
        )

    def decide(
        self,
        state: ReasoningState,
        *,
        is_pure_conversation: bool,
        has_explicit_action_cue: bool,
        has_active_goal: bool,
        force_plugins_for_notes: bool,
    ) -> ExecutiveDecision:
        text_intent = (state.user_intent or "").lower()

        if not state.topic and state.confidence < 0.35 and not has_explicit_action_cue:
            return ExecutiveDecision(
                action="ask_clarification",
                reason="low_confidence_ambiguous",
                store_memory=False,
                clarification_question="I want to help accurately. Do you want an explanation, a tool action, or a search?",
            )

        if len(state.user_intent.strip()) == 0:
            return ExecutiveDecision(
                action="ignore",
                reason="empty_intent",
                store_memory=False,
            )

        if force_plugins_for_notes:
            return ExecutiveDecision(action="use_plugin", reason="note_followup", store_memory=True)

        if has_active_goal or has_explicit_action_cue:
            return ExecutiveDecision(action="use_plugin", reason="goal_or_action_cue", store_memory=True)

        if any(text_intent.startswith(domain) for domain in self.TOOL_DOMAINS):
            if state.confidence >= 0.60:
                return ExecutiveDecision(action="use_plugin", reason="tool_domain_confident", store_memory=True)
            return ExecutiveDecision(
                action="ask_clarification",
                reason="tool_domain_low_confidence",
                store_memory=False,
                clarification_question="Should I perform an action for this, or do you want a general explanation?",
            )

        if is_pure_conversation:
            return ExecutiveDecision(action="use_llm", reason="pure_conversation", store_memory=True)

        return ExecutiveDecision(action="use_llm", reason="default_reasoning", store_memory=True)

    def format_reasoning_state(self, state: ReasoningState) -> str:
        """Render compact, non-CoT internal state for system context."""
        lines = [
            "Internal reasoning state (system-only):",
            f"- user_intent: {state.user_intent}",
            f"- topic: {state.topic or 'unknown'}",
            f"- confidence: {state.confidence:.2f}",
            f"- conversation_goal: {state.conversation_goal or 'general_assistance'}",
            f"- user_goal: {state.user_goal or 'none'}",
            f"- depth_level: {state.depth_level}",
        ]
        if state.plan:
            lines.append(f"- plan: {' | '.join(state.plan)}")
        return "\n".join(lines)

    def _derive_plan(self, intent: str, topic: str, depth_level: int, user_input: str) -> List[str]:
        lowered_intent = (intent or "").lower()
        lowered_input = (user_input or "").lower()

        if lowered_intent.startswith("learning:") or lowered_intent.startswith("question:"):
            if depth_level >= 3 or "example" in lowered_input or "code" in lowered_input:
                return [
                    "explain succinctly",
                    "give concrete example",
                    "offer deeper explanation",
                ]
            return [
                "explain simply",
                "give example",
                "offer deeper explanation",
            ]

        if topic:
            return ["answer current question", "keep topic continuity"]

        return ["answer directly", "ask clarification if ambiguity remains"]


_executive_controller: ExecutiveController | None = None


def get_executive_controller() -> ExecutiveController:
    global _executive_controller
    if _executive_controller is None:
        _executive_controller = ExecutiveController()
    return _executive_controller
