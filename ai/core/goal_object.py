"""Normalized goal object used across executive, action, and turn-state layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Goal:
    goal_id: str
    title: str
    kind: str
    status: str
    parent_goal_id: Optional[str] = None
    blockers: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    next_action: str = ""
    confidence: float = 0.0
    autonomy_level: str = "assisted"
    source_turn: str = ""
    last_result: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_id": str(self.goal_id or ""),
            "title": str(self.title or ""),
            "kind": str(self.kind or "task"),
            "status": str(self.status or "active"),
            "parent_goal_id": str(self.parent_goal_id or "") or None,
            "blockers": [str(x) for x in list(self.blockers or []) if str(x).strip()],
            "success_criteria": [
                str(x) for x in list(self.success_criteria or []) if str(x).strip()
            ],
            "next_action": str(self.next_action or ""),
            "confidence": max(0.0, min(1.0, float(self.confidence or 0.0))),
            "autonomy_level": str(self.autonomy_level or "assisted"),
            "source_turn": str(self.source_turn or ""),
            "last_result": dict(self.last_result or {}),
        }


def goal_from_any(raw: Any) -> Goal:
    """Best-effort normalization from runtime goal-like values to Goal."""
    if isinstance(raw, Goal):
        return raw

    if isinstance(raw, dict):
        goal_id = str(raw.get("goal_id") or raw.get("id") or "").strip()
        title = str(
            raw.get("title")
            or raw.get("description")
            or raw.get("name")
            or goal_id
            or "goal"
        ).strip()
        kind = str(raw.get("kind") or raw.get("intent") or "task").strip()
        status = str(raw.get("status") or "active").strip().lower()
        parent_goal_id = str(raw.get("parent_goal_id") or raw.get("parent_id") or "").strip() or None
        blockers = [str(x) for x in list(raw.get("blockers") or []) if str(x).strip()]
        success_criteria = [
            str(x)
            for x in list(raw.get("success_criteria") or raw.get("criteria") or [])
            if str(x).strip()
        ]
        next_action = str(raw.get("next_action") or raw.get("recommended_next_action") or "").strip()
        confidence = float(raw.get("confidence") or 0.0)
        autonomy_level = str(raw.get("autonomy_level") or "assisted").strip()
        source_turn = str(raw.get("source_turn") or "").strip()
        last_result = raw.get("last_result") if isinstance(raw.get("last_result"), dict) else {}
        if not goal_id:
            goal_id = f"goal::{title[:32]}" if title else "goal::unknown"
        return Goal(
            goal_id=goal_id,
            title=title or "goal",
            kind=kind or "task",
            status=status or "active",
            parent_goal_id=parent_goal_id,
            blockers=blockers,
            success_criteria=success_criteria,
            next_action=next_action,
            confidence=max(0.0, min(1.0, confidence)),
            autonomy_level=autonomy_level or "assisted",
            source_turn=source_turn,
            last_result=dict(last_result or {}),
        )

    goal_id = str(getattr(raw, "goal_id", "") or getattr(raw, "id", "") or "").strip()
    title = str(
        getattr(raw, "title", "")
        or getattr(raw, "description", "")
        or getattr(raw, "name", "")
        or goal_id
        or "goal"
    ).strip()
    kind = str(getattr(raw, "kind", "") or getattr(raw, "intent", "") or "task").strip()
    status = str(getattr(raw, "status", "active") or "active").strip().lower()
    parent_goal_id = str(
        getattr(raw, "parent_goal_id", "") or getattr(raw, "parent_id", "") or ""
    ).strip() or None
    blockers = [str(x) for x in list(getattr(raw, "blockers", []) or []) if str(x).strip()]
    success_criteria = [
        str(x)
        for x in list(getattr(raw, "success_criteria", []) or [])
        if str(x).strip()
    ]
    next_action = str(getattr(raw, "next_action", "") or "").strip()
    confidence = float(getattr(raw, "confidence", 0.0) or 0.0)
    autonomy_level = str(getattr(raw, "autonomy_level", "assisted") or "assisted").strip()
    source_turn = str(getattr(raw, "source_turn", "") or "").strip()
    last_result = getattr(raw, "last_result", {}) if isinstance(getattr(raw, "last_result", {}), dict) else {}

    if not goal_id:
        goal_id = f"goal::{title[:32]}" if title else "goal::unknown"

    return Goal(
        goal_id=goal_id,
        title=title or "goal",
        kind=kind or "task",
        status=status or "active",
        parent_goal_id=parent_goal_id,
        blockers=blockers,
        success_criteria=success_criteria,
        next_action=next_action,
        confidence=max(0.0, min(1.0, confidence)),
        autonomy_level=autonomy_level or "assisted",
        source_turn=source_turn,
        last_result=dict(last_result or {}),
    )
