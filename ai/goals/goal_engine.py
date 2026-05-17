from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_GOALS_FILE = Path("data/goals/goal_stack.json")

_INTENT_PHRASES = (
    "i want to", "i need to", "i'm trying to", "im trying to",
    "my goal is", "i'd like to", "i would like to",
    "we should", "let's build", "let's implement", "help me build",
    "help me create", "help me implement",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Goal:
    goal_id: str = field(default_factory=lambda: f"g_{uuid.uuid4().hex[:8]}")
    description: str = ""
    priority: int = 2  # 1=high 2=medium 3=low
    status: str = "active"  # active | paused | completed | abandoned
    blockers: List[str] = field(default_factory=list)
    next_action: str = ""
    context: str = ""
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    last_worked_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Goal:
        valid = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in valid})

    def days_since_worked(self) -> Optional[float]:
        ts = self.last_worked_at or self.updated_at
        if not ts:
            return None
        try:
            delta = datetime.now(timezone.utc) - datetime.fromisoformat(ts)
            return delta.total_seconds() / 86400
        except Exception:
            return None


class GoalEngine:
    """Persistent cross-session goal stack."""

    def __init__(self, goals_file: Path = _GOALS_FILE):
        self._file = goals_file
        self._goals: List[Goal] = self._load()

    def _load(self) -> List[Goal]:
        if self._file.exists():
            try:
                return [
                    Goal.from_dict(g)
                    for g in json.loads(self._file.read_text(encoding="utf-8"))
                ]
            except Exception:
                pass
        return []

    def _save(self) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        self._file.write_text(
            json.dumps([g.to_dict() for g in self._goals], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def add(
        self,
        description: str,
        *,
        priority: int = 2,
        context: str = "",
        next_action: str = "",
    ) -> Goal:
        goal = Goal(
            description=description,
            priority=priority,
            context=context,
            next_action=next_action,
        )
        self._goals.append(goal)
        self._save()
        return goal

    def update(self, goal_id: str, **fields) -> Optional[Goal]:
        for g in self._goals:
            if g.goal_id == goal_id:
                for k, v in fields.items():
                    if hasattr(g, k):
                        setattr(g, k, v)
                g.updated_at = _now_iso()
                self._save()
                return g
        return None

    def mark_worked(self, goal_id: str) -> None:
        self.update(goal_id, last_worked_at=_now_iso())

    def complete(self, goal_id: str) -> None:
        self.update(goal_id, status="completed")

    def active(self) -> List[Goal]:
        return sorted(
            [g for g in self._goals if g.status == "active"],
            key=lambda g: g.priority,
        )

    def top_goal(self) -> Optional[Goal]:
        goals = self.active()
        return goals[0] if goals else None

    def session_summary(self) -> str:
        goals = self.active()
        if not goals:
            return ""
        lines: List[str] = []
        for g in goals[:4]:
            blocker = f" — blocked: {g.blockers[0]}" if g.blockers else ""
            next_a = f"\n  -> {g.next_action}" if g.next_action else ""
            lines.append(f"[{g.priority}] {g.description}{next_a}{blocker}")
        return "\n".join(lines)

    def stale_goals(self, days: float = 3.0) -> List[Goal]:
        return [
            g for g in self.active()
            if (g.days_since_worked() or 0) >= days
        ]

    def extract_from_text(self, text: str) -> Optional[str]:
        lower = text.lower()
        for phrase in _INTENT_PHRASES:
            if phrase in lower:
                idx = lower.index(phrase) + len(phrase)
                snippet = text[idx: idx + 120].strip()
                return snippet.split(".")[0].split("?")[0].strip() or None
        return None

    def sync_from_active_goals(self, active_goals: List[str]) -> None:
        existing = {g.description.lower() for g in self._goals}
        changed = False
        for desc in active_goals:
            if desc and desc.lower() not in existing:
                goal = Goal(description=desc, priority=2)
                self._goals.append(goal)
                existing.add(desc.lower())
                changed = True
        if changed:
            self._save()


_engine: Optional[GoalEngine] = None


def get_goal_engine() -> GoalEngine:
    global _engine
    if _engine is None:
        _engine = GoalEngine()
    return _engine
