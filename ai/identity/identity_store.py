"""SQLite store for ALICE's persistent opinions and session history.

Opens data/memory/alice.db alongside the existing memories table from
Foundation 1. Two tables: alice_opinions (growing stance library) and
alice_sessions (per-session lifecycle records).
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_DB_PATH = Path("data/memory/alice.db")
_lock = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _confidence(evidence_count: int) -> float:
    return min(0.98, 0.5 + 0.1 * math.log2(1 + max(1, evidence_count)))


class IdentityStore:
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
                CREATE TABLE IF NOT EXISTS alice_opinions (
                    topic          TEXT PRIMARY KEY,
                    stance         TEXT NOT NULL,
                    source         TEXT NOT NULL DEFAULT 'auto',
                    confidence     REAL NOT NULL DEFAULT 0.5,
                    evidence_count INTEGER NOT NULL DEFAULT 1,
                    first_seen     TEXT NOT NULL,
                    last_updated   TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_opinions_confidence
                    ON alice_opinions (confidence DESC, evidence_count DESC);

                CREATE TABLE IF NOT EXISTS alice_sessions (
                    session_id        TEXT PRIMARY KEY,
                    started_at        TEXT NOT NULL,
                    ended_at          TEXT,
                    summary           TEXT,
                    turn_count        INTEGER DEFAULT 0,
                    avg_success_score REAL DEFAULT 0.5,
                    failure_kinds     TEXT DEFAULT '{}',
                    intent_mix        TEXT DEFAULT '{}',
                    goals_touched     TEXT DEFAULT '[]'
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_started
                    ON alice_sessions (started_at DESC);
            """)

    # ------------------------------------------------------------------ opinions

    def upsert_opinion(self, topic: str, stance: str, source: str = "auto") -> None:
        topic = topic.lower().strip()[:120]
        stance = stance.strip()[:200]
        now = _now_iso()
        with _lock, self._conn() as conn:
            row = conn.execute(
                "SELECT evidence_count FROM alice_opinions WHERE topic = ?", (topic,)
            ).fetchone()
            if row:
                new_count = int(row["evidence_count"]) + 1
                conn.execute(
                    """UPDATE alice_opinions
                       SET stance=?, source=?, confidence=?, evidence_count=?, last_updated=?
                       WHERE topic=?""",
                    (stance, source, _confidence(new_count), new_count, now, topic),
                )
            else:
                conn.execute(
                    """INSERT INTO alice_opinions
                       (topic, stance, source, confidence, evidence_count, first_seen, last_updated)
                       VALUES (?, ?, ?, ?, 1, ?, ?)""",
                    (topic, stance, source, _confidence(1), now, now),
                )

    def get_top_opinions(self, n: int = 6) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT topic, stance, source, confidence, evidence_count, last_updated
                   FROM alice_opinions
                   ORDER BY confidence DESC, evidence_count DESC
                   LIMIT ?""",
                (n,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ sessions

    def begin_session(self, session_id: str, started_at: str) -> None:
        with _lock, self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO alice_sessions (session_id, started_at) VALUES (?, ?)",
                (session_id, started_at),
            )

    def end_session(
        self,
        session_id: str,
        ended_at: str,
        summary: str,
        turn_count: int,
        avg_success_score: float,
        failure_kinds: Dict[str, int],
        intent_mix: Dict[str, int],
        goals_touched: Optional[List[str]] = None,
    ) -> None:
        with _lock, self._conn() as conn:
            conn.execute(
                """UPDATE alice_sessions
                   SET ended_at=?, summary=?, turn_count=?, avg_success_score=?,
                       failure_kinds=?, intent_mix=?, goals_touched=?
                   WHERE session_id=?""",
                (
                    ended_at,
                    summary,
                    turn_count,
                    avg_success_score,
                    json.dumps(failure_kinds),
                    json.dumps(intent_mix),
                    json.dumps(list(goals_touched or [])),
                    session_id,
                ),
            )

    def get_last_session(self) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                """SELECT * FROM alice_sessions
                   WHERE ended_at IS NOT NULL
                   ORDER BY started_at DESC
                   LIMIT 1"""
            ).fetchone()
        return dict(row) if row else None


_store: Optional[IdentityStore] = None


def get_identity_store() -> IdentityStore:
    global _store
    if _store is None:
        _store = IdentityStore()
    return _store
