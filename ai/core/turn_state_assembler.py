"""Build hidden turn state summaries for routing and planning."""

from __future__ import annotations

from typing import Any, Dict, List


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _coerce_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


class TurnStateAssembler:
    """Assemble a normalized, hidden state snapshot used for each turn."""

    def __init__(self, world_memory: Any | None = None):
        self.world_memory = world_memory

    def _load_world_state(self) -> Dict[str, Any]:
        if self.world_memory and hasattr(self.world_memory, "load_world_state"):
            try:
                state = self.world_memory.load_world_state() or {}
                if isinstance(state, dict):
                    return dict(state)
            except Exception:
                return {}
        if self.world_memory and hasattr(self.world_memory, "snapshot"):
            try:
                state = self.world_memory.snapshot() or {}
                if isinstance(state, dict):
                    return dict(state)
            except Exception:
                return {}
        return {}

    def _recent_events(self, limit: int = 3) -> List[str]:
        if not self.world_memory or not hasattr(self.world_memory, "memory"):  # pragma: no cover - depends on runtime object
            return []
        events: List[str] = []
        try:
            for item in self.world_memory.memory[-limit:]:
                msg = item.get("message")
                if isinstance(msg, str) and msg.strip():
                    events.append(msg.strip())
        except Exception:
            return []
        return events[-limit:]

    def _confidence_state(self, world_state: Dict[str, Any], action_context: Dict[str, Any]) -> Dict[str, Any]:
        confidence = action_context.get("confidence")
        if confidence is None:
            confidence = world_state.get("last_confidence", 0.5)
        return {
            "confidence": max(0.0, min(1.0, float(confidence) if isinstance(confidence, (int, float)) else 0.5)),
            "ambiguous": bool(action_context.get("ambiguity_detected") or world_state.get("ambiguity_detected")),
        }

    def _question_state(self, world_state: Dict[str, Any], action_context: Dict[str, Any]) -> Dict[str, Any]:
        open_questions = _coerce_list(world_state.get("open_questions"))
        pending_slot = action_context.get("pending_slot")
        return {
            "count": len(open_questions),
            "open": open_questions[:5],
            "pending_slot": pending_slot if isinstance(pending_slot, dict) else {},
            "has_pending_slot": bool(pending_slot),
        }

    def _goal_progress(self, world_state: Dict[str, Any], action_context: Dict[str, Any]) -> Dict[str, Any]:
        goals = _coerce_list(world_state.get("active_goals"))
        completed = _safe_int(world_state.get("goals_completed"), 0)
        return {
            "active_count": len(goals),
            "active": goals[:5],
            "completed": completed,
            "priority_goal": action_context.get("goal") or (goals[0] if goals else ""),
        }

    def _risk_state(self, action_context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "risk_level": str(action_context.get("risk_level") or "low").lower(),
            "requires_simulation": bool(action_context.get("requires_simulation")),
            "action_type": str(action_context.get("action_type") or ""),
        }

    def build(
        self,
        *,
        user_input: str,
        intent: str,
        action_context: Dict[str, Any] | None = None,
        before_state: Dict[str, Any] | None = None,
        extra: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Produce a hidden summary snapshot for this turn."""
        action_context = dict(action_context or {})
        world_state = dict(before_state or self._load_world_state())

        snapshot: Dict[str, Any] = {
            "intent": intent,
            "input_preview": (user_input or "")[:180],
            "confidence_state": self._confidence_state(world_state, action_context),
            "question_state": self._question_state(world_state, action_context),
            "goal_progress": self._goal_progress(world_state, action_context),
            "risk_state": self._risk_state(action_context),
            "recent_events": self._recent_events(limit=3),
        }

        if extra:
            snapshot["extra"] = dict(extra)
        return snapshot

    def to_text(self, snapshot: Dict[str, Any]) -> str:
        """Render a concise hidden summary for prompt/routing context."""
        conf = snapshot.get("confidence_state", {})
        q = snapshot.get("question_state", {})
        g = snapshot.get("goal_progress", {})
        r = snapshot.get("risk_state", {})
        events = snapshot.get("recent_events", []) or []

        lines = [
            "[Hidden Situation Summary]",
            f"Intent: {snapshot.get('intent', '')}",
            f"Confidence: {conf.get('confidence', 0.5):.2f} | Ambiguous: {conf.get('ambiguous', False)}",
            f"Open questions: {q.get('count', 0)} | Pending slot: {q.get('has_pending_slot', False)}",
            f"Active goals: {g.get('active_count', 0)} | Completed: {g.get('completed', 0)}",
            f"Risk: {r.get('risk_level', 'low')} | Simulation required: {r.get('requires_simulation', False)}",
        ]
        if events:
            lines.append("Recent events: " + " | ".join(str(e) for e in events[:3]))
        return "\n".join(lines)
