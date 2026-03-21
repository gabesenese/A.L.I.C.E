"""Dialogue state machine with guarded transitions and clarifying timeout."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Set


class DialogueState(str, Enum):
    READY = "ready"
    CLARIFYING = "clarifying"
    EXECUTING = "executing"
    LEARNING = "learning"


_ALLOWED: Dict[DialogueState, Set[DialogueState]] = {
    DialogueState.READY: {DialogueState.CLARIFYING, DialogueState.EXECUTING, DialogueState.LEARNING},
    DialogueState.CLARIFYING: {DialogueState.READY, DialogueState.EXECUTING, DialogueState.LEARNING, DialogueState.CLARIFYING},
    DialogueState.EXECUTING: {DialogueState.READY, DialogueState.CLARIFYING, DialogueState.LEARNING, DialogueState.EXECUTING},
    DialogueState.LEARNING: {DialogueState.READY, DialogueState.CLARIFYING, DialogueState.EXECUTING, DialogueState.LEARNING},
}


@dataclass
class DialogueStateMachine:
    max_clarifying_turns: int = 5
    state: DialogueState = DialogueState.READY
    clarifying_turns: int = 0

    def transition(self, target: DialogueState, *, reason: str = "") -> bool:
        target = DialogueState(target)
        if target not in _ALLOWED.get(self.state, set()):
            return False

        if target == DialogueState.CLARIFYING:
            self.clarifying_turns += 1
        else:
            self.clarifying_turns = 0

        self.state = target
        if self.state == DialogueState.CLARIFYING and self.clarifying_turns >= self.max_clarifying_turns:
            self.state = DialogueState.READY
            self.clarifying_turns = 0
        return True

    def observe_intent(self, intent: str) -> DialogueState:
        intent = (intent or "").lower()
        if intent == "conversation:clarification_needed":
            self.transition(DialogueState.CLARIFYING, reason="clarification_intent")
        elif any(intent.startswith(prefix) for prefix in ("notes:", "email:", "calendar:", "file_operations:", "reminder:", "system:", "weather:", "time:")):
            self.transition(DialogueState.EXECUTING, reason="tool_intent")
        elif intent.startswith("learning:"):
            self.transition(DialogueState.LEARNING, reason="learning_intent")
        else:
            self.transition(DialogueState.READY, reason="default_ready")
        return self.state

    def as_dict(self) -> Dict[str, str | int]:
        return {
            "state": self.state.value,
            "clarifying_turns": int(self.clarifying_turns),
            "max_clarifying_turns": int(self.max_clarifying_turns),
        }
