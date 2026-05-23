from __future__ import annotations

import json
import re
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

_VAGUE_GOAL = re.compile(
    r"^(?:know|learn|understand|build|make|do|try|work|get|have|use|see|find|"
    r"figure\s+out|think\s+about)\s*(?:it|one|something|this|that|them|things?|"
    r"stuff|more|better|how|what|why)?\.?$",
    re.IGNORECASE,
)

_DANGLING_PRONOUN = re.compile(
    r"\b(?:it|one|this|that|them|something|stuff|things?)\s*$",
    re.IGNORECASE,
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
    milestones: List[Dict[str, Any]] = field(default_factory=list)
    # Each milestone: {"id": str, "text": str, "done": bool, "completed_at": str|None}
    dependencies: List[str] = field(default_factory=list)
    # List of goal_ids that must be completed before this goal is "ready"

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

    def progress_pct(self) -> float:
        """Return 0.0–1.0 based on completed milestones. 0.0 if no milestones."""
        if not self.milestones:
            return 0.0
        done = sum(1 for m in self.milestones if m.get("done"))
        return round(done / len(self.milestones), 2)


class GoalEngine:
    """Persistent cross-session goal stack."""

    def __init__(self, goals_file: Path = _GOALS_FILE):
        self._file = goals_file
        # Foundation 3 — SQLite store (write-through; JSON kept for compatibility)
        try:
            from ai.goals.goal_store import get_goal_store
            self._store = get_goal_store()
            self._store.migrate_from_json(self._file)
        except Exception:
            self._store = None
        self._goals: List[Goal] = self._load()

    def _load(self) -> List[Goal]:
        # Prefer SQLite; fall back to JSON
        if self._store is not None:
            try:
                rows = self._store.load_goals()
                if rows:
                    return [Goal.from_dict(r) for r in rows]
            except Exception:
                pass
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
        if self._store is not None:
            try:
                for g in self._goals:
                    self._store.upsert_goal(g)
            except Exception:
                pass

    def _record_event(
        self, goal_id: str, event_type: str,
        session_id: str = "", intent: str = "", note: str = "",
    ) -> None:
        if self._store is not None:
            try:
                self._store.record_event(goal_id, event_type, session_id, intent, note)
            except Exception:
                pass

    def add(
        self,
        description: str,
        *,
        priority: int = 2,
        context: str = "",
        next_action: str = "",
    ) -> Optional[Goal]:
        desc = str(description or "").strip()
        if len(desc) < 12 or _VAGUE_GOAL.match(desc) or _DANGLING_PRONOUN.search(desc):
            return None
        lower = desc.lower()
        for g in self._goals:
            if g.description.lower() == lower:
                return g
        goal = Goal(
            description=desc,
            priority=priority,
            context=context,
            next_action=next_action,
        )
        self._goals.append(goal)
        self._save()
        self._record_event(goal.goal_id, "created", note=desc[:80])
        # Auto-decompose milestones from the description
        try:
            from ai.planning.action_planner import get_action_planner
            class _Req:
                def __init__(self, g): self.goal = g; self.plan_steps = None
            steps = get_action_planner().decompose(_Req(desc))
            if len(steps) > 1:
                self.populate_milestones_from_steps(goal.goal_id, steps)
        except Exception:
            pass
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

    def mark_worked(
        self, goal_id: str, session_id: str = "", intent: str = ""
    ) -> None:
        self.update(goal_id, last_worked_at=_now_iso())
        self._record_event(goal_id, "worked", session_id=session_id, intent=intent)

    def complete(self, goal_id: str, session_id: str = "", note: str = "") -> None:
        self.update(goal_id, status="completed")
        self._record_event(goal_id, "completed", session_id=session_id, note=note)

    def active(self) -> List[Goal]:
        return sorted(
            [g for g in self._goals if g.status == "active"],
            key=lambda g: g.priority,
        )

    def get_active_goals(self) -> List[Goal]:
        return self.active()

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

    def get_ready_goals(self) -> List[Goal]:
        """Return active goals whose all dependency goals are completed."""
        completed_ids = {g.goal_id for g in self._goals if g.status == "completed"}
        return [
            g for g in self.active()
            if all(dep in completed_ids for dep in g.dependencies)
        ]

    def get_blocker_goals(self, goal_id: str) -> List[Goal]:
        """Return active goals that are blocking the given goal (incomplete dependencies)."""
        completed_ids = {g.goal_id for g in self._goals if g.status == "completed"}
        for g in self._goals:
            if g.goal_id == goal_id:
                return [
                    b for b in self._goals
                    if b.goal_id in g.dependencies and b.goal_id not in completed_ids
                ]
        return []

    def extract_from_text(self, text: str) -> Optional[str]:
        lower = text.lower()
        for phrase in _INTENT_PHRASES:
            if phrase in lower:
                idx = lower.index(phrase) + len(phrase)
                snippet = text[idx: idx + 120].strip()
                candidate = snippet.split(".")[0].split("?")[0].strip()
                if not candidate or len(candidate) < 12:
                    continue
                if _VAGUE_GOAL.match(candidate):
                    continue
                if _DANGLING_PRONOUN.search(candidate):
                    continue
                return candidate
        return None

    def sync_from_active_goals(self, active_goals: List[str]) -> None:
        existing = {g.description.lower() for g in self._goals}
        changed = False
        for desc in active_goals:
            if not desc or len(desc) < 12:
                continue
            if _VAGUE_GOAL.match(desc) or _DANGLING_PRONOUN.search(desc):
                continue
            if desc.lower() not in existing:
                goal = Goal(description=desc, priority=2)
                self._goals.append(goal)
                existing.add(desc.lower())
                changed = True
        if changed:
            self._save()

    def add_milestone(self, goal_id: str, text: str) -> Optional[Dict[str, Any]]:
        """Append a milestone to the goal. Returns the milestone dict or None."""
        text = str(text or "").strip()
        if not text:
            return None
        for g in self._goals:
            if g.goal_id == goal_id:
                milestone = {
                    "id": f"m_{uuid.uuid4().hex[:6]}",
                    "text": text[:120],
                    "done": False,
                    "completed_at": None,
                }
                g.milestones.append(milestone)
                g.updated_at = _now_iso()
                self._save()
                return milestone
        return None

    def complete_milestone(self, goal_id: str, milestone_id: str) -> bool:
        """Mark a milestone as done. Returns True if found and updated."""
        for g in self._goals:
            if g.goal_id == goal_id:
                for m in g.milestones:
                    if m.get("id") == milestone_id and not m.get("done"):
                        m["done"] = True
                        m["completed_at"] = _now_iso()
                        g.last_worked_at = _now_iso()
                        g.updated_at = _now_iso()
                        # Auto-complete goal when all milestones done
                        if all(m2.get("done") for m2 in g.milestones):
                            g.status = "completed"
                        self._save()
                        self._record_event(goal_id, "milestone_advanced", note=m.get("text", "")[:60])
                        return True
        return False

    def populate_milestones_from_steps(self, goal_id: str, steps: List[str]) -> int:
        """Populate a goal's milestones from a decomposed step list. Returns count added."""
        added = 0
        for g in self._goals:
            if g.goal_id == goal_id and not g.milestones:
                for step in steps[:8]:
                    m = self.add_milestone(goal_id, step)
                    if m:
                        added += 1
                break
        return added

    def ingest_completed_intent(self, *, intent: str, user_input: str) -> List[str]:
        """Mark active goals as worked when a completed intent matches their next_action.

        Also advances the first matching incomplete milestone.
        Returns list of goal_ids that were advanced.
        """
        intent_lower = str(intent or "").lower()
        input_lower = str(user_input or "").lower()
        advanced: List[str] = []

        for goal in self.active():
            next_act = str(goal.next_action or "").lower().strip()
            desc = goal.description.lower()

            overlap_with_next = next_act and (
                next_act in intent_lower
                or intent_lower in next_act
                or any(w in intent_lower for w in next_act.split() if len(w) > 4)
            )
            overlap_with_desc = any(
                w in input_lower for w in desc.split() if len(w) > 5
            )

            if overlap_with_next or (overlap_with_desc and not next_act):
                self.mark_worked(goal.goal_id)
                advanced.append(goal.goal_id)
                # Advance first incomplete milestone that matches the intent/input
                for m in goal.milestones:
                    if not m.get("done"):
                        m_text = str(m.get("text") or "").lower()
                        if any(w in m_text for w in intent_lower.split() if len(w) > 4) or \
                           any(w in m_text for w in input_lower.split() if len(w) > 5):
                            self.complete_milestone(goal.goal_id, m["id"])
                        break  # only advance one milestone per turn

        return advanced


_engine: Optional[GoalEngine] = None


def get_goal_engine() -> GoalEngine:
    global _engine
    if _engine is None:
        _engine = GoalEngine()
    return _engine
