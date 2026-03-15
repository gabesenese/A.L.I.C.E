"""
Goal Tracker.

Provides four capabilities built on top of the existing goal fields in
ConversationState:

1. Goal steering   — active goal injected into system prompt as constraint
2. Goal persistence — prevents resets when topic shifts stay related
3. Drift correction — evolves goal instead of replacing it on minor shifts
4. Completion detection — signals when user objective has been met and
                          suggests the natural next step
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SubGoal:
    description: str
    completed: bool = False


@dataclass
class GoalProgress:
    goal_description: str
    subgoals: List[SubGoal] = field(default_factory=list)
    completion_checklist: Dict[str, bool] = field(default_factory=dict)
    progress_score: float = 0.0
    completion_signals: List[str] = field(default_factory=list)
    turns_since_set: int = 0

    def is_complete(self) -> bool:
        return self.progress_score >= 0.85

    def as_dict(self) -> Dict[str, Any]:
        return {
            "goal_description": self.goal_description,
            "subgoals": [
                {"desc": sg.description, "done": sg.completed}
                for sg in self.subgoals
            ],
            "completion_checklist": dict(self.completion_checklist),
            "progress_score": round(self.progress_score, 3),
            "completion_signals": list(self.completion_signals),
            "turns_since_set": self.turns_since_set,
        }


class GoalTracker:
    """
    Tracks active user goal across turns with drift-resilient persistence.

    Design rules:
    - A goal is *replaced* only when the new goal has < 0.30 token overlap
      with the current goal AND has been active for > 2 turns (early pivots
      are treated as subgoals).
    - A goal is *evolved* (subgoal recorded) for minor shifts.
    - Completion is scored incrementally via user acknowledgement tokens
      and response-goal token overlap.
    """

    _COMPLETION_MARKERS = frozenset((
        "thank", "thanks", "perfect", "got it", "makes sense", "that works",
        "solved", "fixed", "done", "great", "awesome", "understood",
        "that's it", "that's what i needed", "exactly", "bingo",
    ))

    _PROGRESS_MARKERS = frozenset((
        "ok", "okay", "and then", "next", "now what",
        "what about", "so", "alright", "sounds good",
    ))

    # Minimum token overlap to consider a new topic goal-related
    _SIMILARITY_THRESHOLD = 0.30

    def __init__(self) -> None:
        self._goal: Optional[GoalProgress] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(
        self,
        user_input: str,
        response: str,
        *,
        user_goal: str,
        conversation_goal: str,
        intent: str,
        current_topic: str = "",
        previous_topic: str = "",
    ) -> None:
        """Process a completed turn and update goal state."""
        raw_goal = (user_goal or "").strip()
        if not raw_goal and conversation_goal:
            raw_goal = conversation_goal.strip()

        if not raw_goal:
            # No new goal signal — keep ticking the existing goal
            if self._goal:
                self._goal.turns_since_set += 1
                self._tick_progress(user_input, response)
            return

        if self._goal and self._goal.goal_description:
            similarity = self._token_similarity(raw_goal, self._goal.goal_description)
            if similarity >= self._SIMILARITY_THRESHOLD:
                # Related topic shift — evolve, do NOT reset
                self._goal.turns_since_set += 1
            else:
                # Sufficiently different topic
                if self._goal.turns_since_set <= 2:
                    # Early divergence — treat as subgoal to preserve context
                    self._goal.subgoals.append(SubGoal(description=raw_goal))
                    self._goal.turns_since_set += 1
                else:
                    # Genuine goal change — replace and reset progress
                    self._goal = GoalProgress(
                        goal_description=raw_goal,
                        completion_checklist=self._build_checklist(raw_goal),
                        turns_since_set=0,
                    )
        else:
            self._goal = GoalProgress(
                goal_description=raw_goal,
                completion_checklist=self._build_checklist(raw_goal),
                turns_since_set=0,
            )

        # Goal evolution: if topic shifts but remains related, create a subgoal.
        if self._goal and current_topic and previous_topic and current_topic != previous_topic:
            related = self._token_similarity(current_topic, self._goal.goal_description) >= self._SIMILARITY_THRESHOLD
            if related:
                subgoal_text = f"advance {current_topic}"
                if not any(sg.description == subgoal_text for sg in self._goal.subgoals):
                    self._goal.subgoals.append(SubGoal(description=subgoal_text))

        self._tick_progress(user_input, response)

    def goal_alignment_score(self, response: str) -> float:
        """
        How well does `response` address the active goal? Returns 0..1.
        Returns 1.0 (no penalty) when no goal is active.
        """
        if not self._goal or not self._goal.goal_description:
            return 1.0
        return self._token_similarity(self._goal.goal_description, response or "")

    def is_goal_achieved(self) -> bool:
        if self._goal is None:
            return False
        # Explicit user acknowledgement is the strongest completion signal
        if "user_acknowledged" in self._goal.completion_signals:
            return True
        if self._goal.completion_checklist and all(self._goal.completion_checklist.values()):
            return True
        return self._goal.is_complete()

    def get_next_step_suggestion(self) -> str:
        """Returns a human-readable suggestion about what to do next."""
        if not self._goal:
            return ""
        if self.is_goal_achieved():
            pending = [sg for sg in self._goal.subgoals if not sg.completed]
            pending_checks = [k for k, done in self._goal.completion_checklist.items() if not done]
            if pending_checks:
                return f"Main goal mostly met. Pending verification: {pending_checks[0].replace('_', ' ')}"
            if pending:
                return f"Main goal met. Still pending: {pending[0].description}"
            return "Goal appears achieved. Ready for the next task."
        progress_pct = int(self._goal.progress_score * 100)
        return (
            f"Progress toward '{self._goal.goal_description}': ~{progress_pct}%. "
            "Continue to advance this goal."
        )

    def get_goal_prompt_injection(self) -> str:
        """
        Returns a steering block injected into the system prompt so the LLM
        stays aligned with the active goal on every turn.
        """
        if not self._goal or not self._goal.goal_description:
            return ""
        lines = [f"Active goal: {self._goal.goal_description}"]
        pending = [sg.description for sg in self._goal.subgoals if not sg.completed]
        if pending:
            lines.append(f"Pending subgoals: {', '.join(pending[:3])}")
        pending_checks = [k for k, done in self._goal.completion_checklist.items() if not done]
        if pending_checks:
            lines.append(f"Pending checks: {', '.join(p.replace('_', ' ') for p in pending_checks[:3])}")
        if self.is_goal_achieved():
            lines.append("Status: goal achieved — offer a summary or next step.")
        else:
            lines.append("Keep response aligned with and advancing this goal.")
        return "\n".join(lines)

    def get_status(self) -> Optional[Dict[str, Any]]:
        """Return current goal state for diagnostics or memory serialization."""
        return self._goal.as_dict() if self._goal else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tick_progress(self, user_input: str, response: str) -> None:
        if not self._goal:
            return
        low_input = (user_input or "").lower()
        low_resp = (response or "").lower()

        # User acknowledged a satisfactory answer
        if any(m in low_input for m in self._COMPLETION_MARKERS):
            if "user_acknowledged" not in self._goal.completion_signals:
                self._goal.completion_signals.append("user_acknowledged")
            self._goal.progress_score = min(1.0, self._goal.progress_score + 0.40)

        # Response addresses the goal
        if self._goal.goal_description:
            overlap = self._token_similarity(self._goal.goal_description, low_resp)
            self._goal.progress_score = min(
                1.0, self._goal.progress_score + (overlap * 0.15)
            )

        # Completion checklist updates for troubleshooting/debug goals.
        checklist = self._goal.completion_checklist
        if checklist:
            if "problem_identified" in checklist and any(k in low_resp for k in ("cause", "problem", "issue", "error")):
                checklist["problem_identified"] = True
            if "solution_given" in checklist and any(k in low_resp for k in ("fix", "solution", "change", "update", "step")):
                checklist["solution_given"] = True
            if "verification_confirmed" in checklist and any(k in low_input for k in ("worked", "fixed", "confirmed", "verified", "done")):
                checklist["verification_confirmed"] = True
            if all(checklist.values()):
                self._goal.progress_score = min(1.0, self._goal.progress_score + 0.40)

        # Complete subgoals that appear covered by the response
        for sg in self._goal.subgoals:
            if not sg.completed and sg.description:
                if self._token_similarity(sg.description, low_resp) >= 0.40:
                    sg.completed = True
                    self._goal.completion_signals.append(f"subgoal:{sg.description[:30]}")
                    self._goal.progress_score = min(
                        1.0, self._goal.progress_score + 0.10
                    )

    def _token_similarity(self, a: str, b: str) -> float:
        ta = set(re.findall(r"[a-z0-9']+", a.lower()))
        tb = set(re.findall(r"[a-z0-9']+", b.lower()))
        if not ta:
            return 0.0
        return len(ta.intersection(tb)) / max(len(ta), 1)

    def _build_checklist(self, goal_text: str) -> Dict[str, bool]:
        low = (goal_text or "").lower()
        if any(k in low for k in ("fix", "bug", "debug", "error", "issue")):
            return {
                "problem_identified": False,
                "solution_given": False,
                "verification_confirmed": False,
            }
        return {}


_goal_tracker: GoalTracker | None = None


def get_goal_tracker() -> GoalTracker:
    global _goal_tracker
    if _goal_tracker is None:
        _goal_tracker = GoalTracker()
    return _goal_tracker
