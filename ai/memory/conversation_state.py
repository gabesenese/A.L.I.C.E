"""
Conversation state tracking for coherent multi-turn dialogue.

Tracks conversational state beyond raw memory retrieval:
- conversation_topic
- conversation_goal
- user_goal
- depth_level
- question_chain
- intent_chain
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class ConversationState:
    conversation_topic: str = ""
    conversation_goal: str = ""
    user_goal: str = ""
    depth_level: int = 0
    question_chain: List[str] = field(default_factory=list)
    intent_chain: List[str] = field(default_factory=list)
    pending_followup_slot: Dict[str, Any] = field(default_factory=dict)
    pending_clarification: Dict[str, Any] = field(default_factory=dict)
    selected_object_reference: str = ""
    paused_autonomy_reason: str = ""
    last_recovery_outcome: str = ""
    last_trigger_goal: str = ""
    last_updated: str = ""


class ConversationStateTracker:
    """Maintains a compact state model across user turns."""

    def __init__(self, max_chain: int = 8, max_depth: int = 5):
        self.max_chain = max(3, int(max_chain))
        self.max_depth = max(3, int(max_depth))
        self._state = ConversationState()

    def update_state(
        self,
        user_input: str,
        intent: str,
        entities: Optional[Dict[str, Any]] = None,
        goal_hint: str = "",
    ) -> Dict[str, Any]:
        entities = entities or {}
        previous_topic = self._state.conversation_topic

        inferred_topic = self._infer_topic(user_input, entities, previous_topic)
        topic_changed = bool(
            inferred_topic and previous_topic and inferred_topic != previous_topic
        )
        if inferred_topic:
            self._state.conversation_topic = inferred_topic

        question_like = self._is_question_like(user_input, intent)

        if topic_changed:
            # Drift correction: only hard-reset chains when the new topic is
            # genuinely unrelated to the current goal/topic.  A related shift
            # (e.g. a sub-question within the same learning thread) evolves
            # depth instead of discarding context.
            _goal_related = self._topic_similar_to_goal(inferred_topic or "")
            if _goal_related:
                # Related shift — deepen, keep chains intact
                if question_like:
                    self._state.depth_level = min(
                        self.max_depth,
                        max(1, self._state.depth_level) + 1,
                    )
            else:
                # Genuine topic change — reset chains
                self._state.question_chain = []
                self._state.intent_chain = []
                self._state.depth_level = 1 if question_like else 0
        elif question_like and self._state.conversation_topic:
            inc = self._depth_increment(user_input)
            self._state.depth_level = min(
                self.max_depth,
                max(1, self._state.depth_level) + inc,
            )

        if question_like:
            normalized_question = " ".join((user_input or "").strip().split())
            if normalized_question:
                if (
                    not self._state.question_chain
                    or self._state.question_chain[-1] != normalized_question
                ):
                    self._state.question_chain.append(normalized_question)
                    self._state.question_chain = self._state.question_chain[
                        -self.max_chain :
                    ]

        if intent:
            self._state.intent_chain.append(intent)
            self._state.intent_chain = self._state.intent_chain[-self.max_chain :]

        self._state.conversation_goal = self._infer_conversation_goal(
            intent=intent,
            user_input=user_input,
            topic=self._state.conversation_topic,
        )

        inferred_user_goal = self._infer_user_goal(user_input, entities, goal_hint)
        if inferred_user_goal:
            self._state.user_goal = inferred_user_goal
        elif (
            self._state.conversation_goal == "learning"
            and self._state.conversation_topic
        ):
            self._state.user_goal = f"understand {self._state.conversation_topic}"

        self._state.last_updated = datetime.now().isoformat()
        return self.get_state_summary()

    def get_state_summary(self) -> Dict[str, Any]:
        return {
            "conversation_topic": self._state.conversation_topic,
            "conversation_goal": self._state.conversation_goal,
            "user_goal": self._state.user_goal,
            "depth_level": self._state.depth_level,
            "question_chain": list(self._state.question_chain),
            "intent_chain": list(self._state.intent_chain),
            "pending_followup_slot": dict(self._state.pending_followup_slot or {}),
            "pending_clarification": dict(self._state.pending_clarification or {}),
            "selected_object_reference": self._state.selected_object_reference,
            "paused_autonomy_reason": self._state.paused_autonomy_reason,
            "last_recovery_outcome": self._state.last_recovery_outcome,
            "last_trigger_goal": self._state.last_trigger_goal,
            "last_updated": self._state.last_updated,
        }

    def format_for_prompt(self) -> str:
        state = self.get_state_summary()
        if not any(
            [
                state["conversation_topic"],
                state["conversation_goal"],
                state["user_goal"],
                state["question_chain"],
            ]
        ):
            return ""

        intent_chain = (
            " -> ".join(state["intent_chain"][-4:]) if state["intent_chain"] else ""
        )
        question_chain = (
            " -> ".join(state["question_chain"][-4:]) if state["question_chain"] else ""
        )

        lines = ["Conversation state:"]
        if state["conversation_topic"]:
            lines.append(f"- topic: {state['conversation_topic']}")
        if state["conversation_goal"]:
            lines.append(f"- conversation_goal: {state['conversation_goal']}")
        if state["user_goal"]:
            lines.append(f"- user_goal: {state['user_goal']}")
        if state["depth_level"]:
            lines.append(f"- depth_level: {state['depth_level']}")
        if intent_chain:
            lines.append(f"- intent_chain: {intent_chain}")
        if question_chain:
            lines.append(f"- question_chain: {question_chain}")
        lines.append(
            "Treat follow-up questions as part of this same chain unless the topic clearly changes."
        )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return self.get_state_summary()

    def set_pending_followup_slot(self, slot_state: Dict[str, Any]) -> None:
        self._state.pending_followup_slot = dict(slot_state or {})
        self._state.last_updated = datetime.now().isoformat()

    def clear_pending_followup_slot(self) -> None:
        self._state.pending_followup_slot = {}
        self._state.last_updated = datetime.now().isoformat()

    def set_pending_clarification(self, pending: Dict[str, Any]) -> None:
        self._state.pending_clarification = dict(pending or {})
        self._state.last_updated = datetime.now().isoformat()

    def clear_pending_clarification(self) -> None:
        self._state.pending_clarification = {}
        self._state.last_updated = datetime.now().isoformat()

    def set_selected_object_reference(self, value: str) -> None:
        self._state.selected_object_reference = str(value or "")
        self._state.last_updated = datetime.now().isoformat()

    def set_autonomy_pause_reason(self, reason: str) -> None:
        self._state.paused_autonomy_reason = str(reason or "")
        self._state.last_updated = datetime.now().isoformat()

    def set_last_recovery_outcome(self, outcome: str) -> None:
        self._state.last_recovery_outcome = str(outcome or "")
        self._state.last_updated = datetime.now().isoformat()

    def set_last_trigger_goal(self, goal_id: str) -> None:
        self._state.last_trigger_goal = str(goal_id or "")
        self._state.last_updated = datetime.now().isoformat()

    def load_state(self, data: Dict[str, Any]) -> None:
        if not isinstance(data, dict):
            return
        self._state = ConversationState(
            conversation_topic=str(data.get("conversation_topic", "") or ""),
            conversation_goal=str(data.get("conversation_goal", "") or ""),
            user_goal=str(data.get("user_goal", "") or ""),
            depth_level=int(data.get("depth_level", 0) or 0),
            question_chain=[
                str(x) for x in data.get("question_chain", [])[-self.max_chain :]
            ],
            intent_chain=[
                str(x) for x in data.get("intent_chain", [])[-self.max_chain :]
            ],
            pending_followup_slot=dict(data.get("pending_followup_slot", {}) or {}),
            pending_clarification=dict(data.get("pending_clarification", {}) or {}),
            selected_object_reference=str(data.get("selected_object_reference", "") or ""),
            paused_autonomy_reason=str(data.get("paused_autonomy_reason", "") or ""),
            last_recovery_outcome=str(data.get("last_recovery_outcome", "") or ""),
            last_trigger_goal=str(data.get("last_trigger_goal", "") or ""),
            last_updated=str(data.get("last_updated", "") or ""),
        )

    def _topic_similar_to_goal(self, new_topic: str) -> bool:
        """
        Return True when the new topic is semantically related to the current
        user_goal or conversation_topic (token overlap >= 0.30).
        Prevents unnecessary chain resets on minor topic shifts.
        """
        anchor = self._state.user_goal or self._state.conversation_topic
        if not anchor or not new_topic:
            return False
        anchor_tokens = set(re.findall(r"[a-z0-9']+", anchor.lower()))
        new_tokens = set(re.findall(r"[a-z0-9']+", new_topic.lower()))
        if not anchor_tokens:
            return False
        overlap = len(anchor_tokens.intersection(new_tokens)) / max(
            len(anchor_tokens), 1
        )
        return overlap >= 0.30

    def _infer_topic(
        self, user_input: str, entities: Dict[str, Any], previous_topic: str
    ) -> str:
        for key in ("topic", "subject", "concept"):
            value = entities.get(key)
            if isinstance(value, str) and value.strip():
                return self._clean_topic(value)
            if isinstance(value, list) and value:
                first = value[0]
                if isinstance(first, str) and first.strip():
                    return self._clean_topic(first)

        text = (user_input or "").strip()
        patterns = [
            r"(?:what is|what's|define|explain|tell me about|learn|study)\s+(.+)$",
            r"(?:example of|show me an example of|show example of)\s+(.+)$",
            r"how does\s+(.+?)\s+work\??$",
        ]
        for pattern in patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return self._clean_topic(m.group(1))

        if previous_topic and self._is_followup_question(text):
            return previous_topic

        return ""

    def _clean_topic(self, raw: str) -> str:
        text = (raw or "").strip().strip(" .?!")
        # Trim common trailing qualifiers to keep the core concept.
        text = re.sub(
            r"\b(?:please|for me|for us)$", "", text, flags=re.IGNORECASE
        ).strip()
        return text[:80]

    def _is_question_like(self, user_input: str, intent: str) -> bool:
        text = (user_input or "").strip().lower()
        if "?" in text:
            return True
        if intent.startswith("question:") or intent in {
            "conversation:question",
            "learning:study_topic",
            "study_topic",
        }:
            return True
        return bool(
            re.match(
                r"^(what|why|how|can|could|would|should|is|are|do|does|did)\b", text
            )
        )

    def _is_followup_question(self, user_input: str) -> bool:
        text = (user_input or "").strip().lower()
        if not text:
            return False
        followup_starts = (
            "why",
            "how",
            "can",
            "could",
            "show",
            "example",
            "what about",
        )
        return text.startswith(followup_starts)

    def _depth_increment(self, user_input: str) -> int:
        text = (user_input or "").lower()
        inc = 1
        if re.search(r"\b(why|how|tradeoff|useful|benefit)\b", text):
            inc += 1
        if re.search(r"\b(example|show|code|implement|practical)\b", text):
            inc += 1
        return min(inc, 2)

    def _infer_conversation_goal(self, intent: str, user_input: str, topic: str) -> str:
        lowered_intent = (intent or "").lower()
        lowered_input = (user_input or "").lower()

        if lowered_intent.startswith("learning:") or lowered_intent.startswith(
            "question:"
        ):
            return "learning"
        if lowered_intent in {"conversation:question", "study_topic"}:
            return "learning"
        if lowered_intent == "conversation:goal_statement":
            return "project_direction"
        if lowered_intent.startswith("notes:"):
            return "note_management"
        if lowered_intent.startswith("email:"):
            return "email_management"
        if lowered_intent.startswith("calendar:"):
            return "scheduling"
        if topic and self._is_question_like(lowered_input, lowered_intent):
            return "learning"
        return "general_assistance"

    def _infer_user_goal(
        self, user_input: str, entities: Dict[str, Any], goal_hint: str
    ) -> str:
        if goal_hint:
            return str(goal_hint).strip()[:120]

        for key in ("goal", "user_goal", "objective"):
            val = entities.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()[:120]

        text = (user_input or "").strip()
        patterns = [
            r"\bi\s+(?:want|need|would like|am trying)\s+to\s+(.+)$",
            r"\bhelp me\s+(.+)$",
            r"\bshow me\s+(.+)$",
        ]
        for pattern in patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return m.group(1).strip(" .?!")[:120]

        return ""


_tracker_instance: Optional[ConversationStateTracker] = None


def get_conversation_state_tracker(
    max_chain: int = 8, max_depth: int = 5
) -> ConversationStateTracker:
    """Get or create a process-wide conversation state tracker."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = ConversationStateTracker(
            max_chain=max_chain, max_depth=max_depth
        )
    return _tracker_instance
