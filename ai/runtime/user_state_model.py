"""Runtime user-state model for canonical execution pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class UserState:
    user_id: str
    current_task: str = ""
    prior_task: str = ""
    unresolved_references: List[str] = field(default_factory=list)
    active_goals: List[str] = field(default_factory=list)
    last_tool_used: str = ""
    last_result_produced: str = ""
    last_route: str = ""
    last_intent: str = ""
    world_state_snapshot: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(
        default_factory=lambda: {
            "response_brevity": "balanced",
            "confirmation_style": "safety_first",
            "risk_tolerance": "medium",
        }
    )
    updated_at: str = ""


class UserStateModel:
    """In-memory user-state registry with per-turn updates."""

    def __init__(self):
        self._states: Dict[str, UserState] = {}

    def get_or_create(self, user_id: str) -> UserState:
        key = str(user_id or "default").strip() or "default"
        state = self._states.get(key)
        if state is None:
            state = UserState(user_id=key)
            self._states[key] = state
        return state

    def update_turn(
        self,
        *,
        user_id: str,
        intent: str,
        route: str,
        unresolved_references: Optional[List[str]] = None,
        active_goals: Optional[List[str]] = None,
        last_tool_used: str = "",
        last_result_produced: str = "",
        world_state_snapshot: Optional[Dict[str, Any]] = None,
    ) -> UserState:
        state = self.get_or_create(user_id)
        state.prior_task = state.current_task
        state.current_task = str(intent or "")
        state.last_route = str(route or "")
        state.last_intent = str(intent or "")
        state.unresolved_references = list(unresolved_references or [])
        if active_goals is not None:
            state.active_goals = list(active_goals)
        if last_tool_used:
            state.last_tool_used = str(last_tool_used)
        if last_result_produced:
            state.last_result_produced = str(last_result_produced)
        if world_state_snapshot is not None:
            state.world_state_snapshot = dict(world_state_snapshot)
        state.updated_at = datetime.utcnow().isoformat()
        return state

    def update_preferences(
        self,
        *,
        user_id: str,
        response_brevity: Optional[str] = None,
        confirmation_style: Optional[str] = None,
        risk_tolerance: Optional[str] = None,
    ) -> UserState:
        """P2 personalization profile updates."""
        state = self.get_or_create(user_id)
        if response_brevity:
            state.preferences["response_brevity"] = str(response_brevity)
        if confirmation_style:
            state.preferences["confirmation_style"] = str(confirmation_style)
        if risk_tolerance:
            state.preferences["risk_tolerance"] = str(risk_tolerance)
        state.updated_at = datetime.utcnow().isoformat()
        return state

    def get_preference_profile(self, user_id: str) -> Dict[str, Any]:
        state = self.get_or_create(user_id)
        return dict(state.preferences)
