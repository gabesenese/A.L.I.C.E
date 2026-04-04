"""Build hidden turn state summaries for routing and planning."""

from __future__ import annotations

from typing import Any, Dict, List

from ai.core.goal_object import goal_from_any


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
        if not self.world_memory or not hasattr(
            self.world_memory, "memory"
        ):  # pragma: no cover - depends on runtime object
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

    def _confidence_state(
        self, world_state: Dict[str, Any], action_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        confidence = action_context.get("confidence")
        if confidence is None:
            confidence = world_state.get("last_confidence", 0.5)
        return {
            "confidence": max(
                0.0,
                min(
                    1.0,
                    float(confidence) if isinstance(confidence, (int, float)) else 0.5,
                ),
            ),
            "ambiguous": bool(
                action_context.get("ambiguity_detected")
                or world_state.get("ambiguity_detected")
            ),
        }

    def _question_state(
        self, world_state: Dict[str, Any], action_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        open_questions = _coerce_list(world_state.get("open_questions"))
        pending_slot = action_context.get("pending_slot")
        return {
            "count": len(open_questions),
            "open": open_questions[:5],
            "pending_slot": pending_slot if isinstance(pending_slot, dict) else {},
            "has_pending_slot": bool(pending_slot),
        }

    def _goal_progress(
        self, world_state: Dict[str, Any], action_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        goals = _coerce_list(world_state.get("active_goals"))
        completed = _safe_int(world_state.get("goals_completed"), 0)
        return {
            "active_count": len(goals),
            "active": goals[:5],
            "completed": completed,
            "priority_goal": action_context.get("goal") or (goals[0] if goals else ""),
        }

    def _goal_state(
        self, world_state: Dict[str, Any], action_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        raw_stack: List[Any] = []

        if action_context.get("goal") is not None:
            raw_stack.append(action_context.get("goal"))

        for key in ("active_goal_stack", "goal_stack", "active_goals"):
            raw_stack.extend(_coerce_list(world_state.get(key)))

        if world_state.get("current_goal"):
            raw_stack.insert(0, world_state.get("current_goal"))

        normalized: List[Dict[str, Any]] = []
        seen = set()
        for raw in raw_stack:
            try:
                goal = goal_from_any(raw).to_dict()
            except Exception:
                continue
            gid = str(goal.get("goal_id") or "").strip()
            if gid in seen:
                continue
            seen.add(gid)
            normalized.append(goal)

        priority = normalized[0] if normalized else {}
        blockers: List[str] = []
        for g in normalized:
            blockers.extend(
                [str(x) for x in list(g.get("blockers") or []) if str(x).strip()]
            )

        dedup_blockers: List[str] = []
        bseen = set()
        for blocker in blockers:
            if blocker in bseen:
                continue
            bseen.add(blocker)
            dedup_blockers.append(blocker)

        return {
            "active_goal_count": len(normalized),
            "active_goal_stack": normalized[:8],
            "priority_goal": dict(priority or {}),
            "blockers": dedup_blockers[:10],
            "next_action": str((priority or {}).get("next_action") or "").strip(),
            "status": str((priority or {}).get("status") or "").strip().lower(),
        }

    def _continuation_state(
        self, world_state: Dict[str, Any], action_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        pending_slot = action_context.get("pending_slot")
        if not isinstance(pending_slot, dict):
            pending_slot = world_state.get("pending_clarification")
        if not isinstance(pending_slot, dict):
            pending_slot = {}

        selected_reference = str(
            action_context.get("selected_reference")
            or world_state.get("selected_object_reference")
            or world_state.get("selected_reference")
            or ""
        ).strip()

        parent_context = {
            "parent_request": str(pending_slot.get("parent_request") or "").strip(),
            "parent_intent": str(pending_slot.get("parent_intent") or "").strip(),
            "slot_type": str(
                pending_slot.get("slot_type") or pending_slot.get("type") or ""
            ).strip(),
        }

        mode = "normal"
        if parent_context["slot_type"]:
            mode = "clarification_followup"
        elif selected_reference:
            mode = "focused_followup"

        return {
            "mode": mode,
            "has_pending_clarification": bool(parent_context["slot_type"]),
            "pending_clarification": dict(pending_slot or {}),
            "parent_context": parent_context,
            "selected_reference": selected_reference,
        }

    def _last_action_state(self, world_state: Dict[str, Any]) -> Dict[str, Any]:
        raw = world_state.get("last_action")
        if isinstance(raw, dict):
            return {
                "plugin": str(raw.get("plugin") or "").strip(),
                "action": str(raw.get("action") or "").strip(),
                "status": str(raw.get("status") or "").strip(),
                "success": bool(raw.get("success", False)),
                "timestamp": str(raw.get("timestamp") or "").strip(),
            }

        return {
            "plugin": str(world_state.get("last_plugin") or "").strip(),
            "action": str(
                world_state.get("last_tool_action")
                or world_state.get("last_action_name")
                or ""
            ).strip(),
            "status": str(world_state.get("last_status") or "").strip(),
            "success": bool(world_state.get("last_success", False)),
            "timestamp": str(world_state.get("last_action_at") or "").strip(),
        }

    def _selected_focus_state(
        self, world_state: Dict[str, Any], action_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        selected_entity = str(
            action_context.get("selected_entity")
            or world_state.get("selected_object_reference")
            or world_state.get("selected_entity")
            or ""
        ).strip()
        topic_focus = str(
            action_context.get("topic_focus")
            or world_state.get("conversation_topic")
            or world_state.get("topic")
            or ""
        ).strip()
        return {
            "selected_entity": selected_entity,
            "topic_focus": topic_focus,
            "has_focus": bool(selected_entity or topic_focus),
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
            "goal_state": self._goal_state(world_state, action_context),
            "goal_progress": self._goal_progress(world_state, action_context),
            "risk_state": self._risk_state(action_context),
            "continuation_state": self._continuation_state(world_state, action_context),
            "last_action_state": self._last_action_state(world_state),
            "selected_focus": self._selected_focus_state(world_state, action_context),
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
        gs = snapshot.get("goal_state", {})
        r = snapshot.get("risk_state", {})
        c = snapshot.get("continuation_state", {})
        a = snapshot.get("last_action_state", {})
        f = snapshot.get("selected_focus", {})
        events = snapshot.get("recent_events", []) or []

        lines = [
            "[Hidden Situation Summary]",
            f"Intent: {snapshot.get('intent', '')}",
            f"Confidence: {conf.get('confidence', 0.5):.2f} | Ambiguous: {conf.get('ambiguous', False)}",
            f"Open questions: {q.get('count', 0)} | Pending slot: {q.get('has_pending_slot', False)}",
            f"Active goals: {g.get('active_count', 0)} | Completed: {g.get('completed', 0)}",
            f"Goal stack: {gs.get('active_goal_count', 0)} | Next action: {gs.get('next_action', '') or 'none'}",
            f"Risk: {r.get('risk_level', 'low')} | Simulation required: {r.get('requires_simulation', False)}",
            f"Continuation mode: {c.get('mode', 'normal')} | Selected ref: {c.get('selected_reference', '') or 'none'}",
            f"Last action: {a.get('plugin', '')}:{a.get('action', '')} | Success: {a.get('success', False)}",
            f"Focus: {f.get('selected_entity', '') or f.get('topic_focus', '') or 'none'}",
        ]
        if events:
            lines.append("Recent events: " + " | ".join(str(e) for e in events[:3]))
        return "\n".join(lines)
