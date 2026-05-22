"""SQLite store for ALICE's goal tracking — Foundation 3.

Opens data/memory/alice.db alongside Foundation 1 (memories) and
Foundation 2 (opinions, sessions). Two tables:
  alice_goals       — full goal state
  alice_goal_events — audit trail per goal (worked, completed, blocked, etc.)
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

_DB_PATH = Path("data/memory/alice.db")
_lock = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class GoalStore:
    def __init__(self, db_path: Path = _DB_PATH) -> None:
        self._path = db_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS alice_goals (
                    goal_id        TEXT PRIMARY KEY,
                    description    TEXT NOT NULL,
                    priority       INTEGER DEFAULT 2,
                    status         TEXT DEFAULT 'active',
                    blockers       TEXT DEFAULT '[]',
                    next_action    TEXT DEFAULT '',
                    context        TEXT DEFAULT '',
                    milestones     TEXT DEFAULT '[]',
                    dependencies   TEXT DEFAULT '[]',
                    created_at     TEXT NOT NULL,
                    updated_at     TEXT NOT NULL,
                    last_worked_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_goals_status
                    ON alice_goals (status, priority);

                CREATE TABLE IF NOT EXISTS alice_goal_events (
                    event_id    TEXT PRIMARY KEY,
                    goal_id     TEXT NOT NULL,
                    event_type  TEXT NOT NULL,
                    session_id  TEXT DEFAULT '',
                    intent      TEXT DEFAULT '',
                    note        TEXT DEFAULT '',
                    occurred_at TEXT NOT NULL,
                    FOREIGN KEY (goal_id) REFERENCES alice_goals(goal_id)
                );
                CREATE INDEX IF NOT EXISTS idx_goal_events_goal
                    ON alice_goal_events (goal_id, occurred_at DESC);
            """)

    # ------------------------------------------------------------------ goals

    def upsert_goal(self, goal: Any) -> None:
        """Write a Goal object to SQLite (INSERT OR REPLACE)."""
        with _lock, self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO alice_goals
                   (goal_id, description, priority, status, blockers, next_action,
                    context, milestones, dependencies, created_at, updated_at, last_worked_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(goal.goal_id),
                    str(goal.description),
                    int(goal.priority),
                    str(goal.status),
                    json.dumps(list(goal.blockers or [])),
                    str(goal.next_action or ""),
                    str(goal.context or ""),
                    json.dumps(list(goal.milestones or [])),
                    json.dumps(list(goal.dependencies or [])),
                    str(goal.created_at),
                    str(goal.updated_at),
                    goal.last_worked_at,
                ),
            )

    def load_goals(self) -> List[Dict[str, Any]]:
        """Return all goals as raw dicts (GoalEngine constructs Goal objects)."""
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM alice_goals").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            for field in ("blockers", "milestones", "dependencies"):
                try:
                    d[field] = json.loads(d[field] or "[]")
                except Exception:
                    d[field] = []
            out.append(d)
        return out

    def delete_goal(self, goal_id: str) -> None:
        with _lock, self._conn() as conn:
            conn.execute("DELETE FROM alice_goals WHERE goal_id = ?", (goal_id,))

    # ------------------------------------------------------------------ events

    def record_event(
        self,
        goal_id: str,
        event_type: str,
        session_id: str = "",
        intent: str = "",
        note: str = "",
    ) -> None:
        with _lock, self._conn() as conn:
            conn.execute(
                """INSERT INTO alice_goal_events
                   (event_id, goal_id, event_type, session_id, intent, note, occurred_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid4()),
                    goal_id,
                    event_type,
                    session_id or "",
                    intent or "",
                    note or "",
                    _now_iso(),
                ),
            )

    def get_recent_events(self, goal_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM alice_goal_events
                   WHERE goal_id = ?
                   ORDER BY occurred_at DESC
                   LIMIT ?""",
                (goal_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ migration

    def migrate_from_json(self, json_path: Path) -> int:
        """One-time import of existing goal_stack.json. Returns count imported."""
        if not json_path.exists():
            return 0
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            return 0

        existing = {r["goal_id"] for r in self.load_goals()}
        count = 0
        for item in raw:
            if not isinstance(item, dict):
                continue
            goal_id = str(item.get("goal_id") or "")
            if not goal_id or goal_id in existing:
                continue

            class _G:
                pass

            g = _G()
            g.goal_id = goal_id
            g.description = str(item.get("description") or "")
            g.priority = int(item.get("priority") or 2)
            g.status = str(item.get("status") or "active")
            g.blockers = list(item.get("blockers") or [])
            g.next_action = str(item.get("next_action") or "")
            g.context = str(item.get("context") or "")
            g.milestones = list(item.get("milestones") or [])
            g.dependencies = list(item.get("dependencies") or [])
            g.created_at = str(item.get("created_at") or _now_iso())
            g.updated_at = str(item.get("updated_at") or _now_iso())
            g.last_worked_at = item.get("last_worked_at")
            if g.description:
                self.upsert_goal(g)
                count += 1
        return count


_store: Optional[GoalStore] = None


def get_goal_store() -> GoalStore:
    global _store
    if _store is None:
        _store = GoalStore()
    return _store
