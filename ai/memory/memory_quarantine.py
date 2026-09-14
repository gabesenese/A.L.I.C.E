from __future__ import annotations

import logging
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, ContextManager, Dict, List, Optional

from ai.memory.memory_store import sqlite_connection

logger = logging.getLogger(__name__)

_DB_PATH = Path("data/memory/alice.db")
_DEFAULT_TTL_DAYS = 7.0


class MemoryQuarantine:
    """
    Quarantine system for low-quality or incorrect memories.

    Memories enter quarantine instead of being deleted immediately.
    After TTL_DAYS they are auto-purged (both quarantine record and the
    underlying memories row) unless a human has reviewed and released them.

    Quarantine triggers (called externally):
      - composite score < LOW_SCORE_THRESHOLD after rescoring
      - user flags memory as incorrect
      - contradiction detected with high confidence
      - failed answer verification

    Schema: quarantine table in alice.db
    """

    LOW_SCORE_THRESHOLD = 0.18

    def __init__(self, db_path: Path = _DB_PATH, ttl_days: float = _DEFAULT_TTL_DAYS) -> None:
        self.db_path = db_path
        self.ttl_days = ttl_days
        self._init_schema()

    def _conn(self, *, immediate: bool = False) -> ContextManager[sqlite3.Connection]:
        return sqlite_connection(self.db_path, immediate=immediate)

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quarantine (
                    id             TEXT PRIMARY KEY,
                    memory_id      TEXT NOT NULL,
                    reason         TEXT NOT NULL,
                    score          REAL,
                    quarantined_at TEXT NOT NULL,
                    expires_at     TEXT NOT NULL,
                    reviewed       INTEGER DEFAULT 0,
                    released       INTEGER DEFAULT 0
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_q_mid ON quarantine(memory_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_q_exp ON quarantine(expires_at)")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _insert(
        self,
        conn: sqlite3.Connection,
        memory_id: str,
        reason: str,
        score: Optional[float],
    ) -> str:
        qid = f"q_{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc)
        expires = now + timedelta(days=self.ttl_days)
        conn.execute(
            """
            INSERT OR REPLACE INTO quarantine
                (id, memory_id, reason, score, quarantined_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (qid, memory_id, reason, score, now.isoformat(), expires.isoformat()),
        )
        return qid

    @staticmethod
    def _held(conn: sqlite3.Connection, memory_id: str) -> bool:
        return (
            conn.execute(
                "SELECT 1 FROM quarantine WHERE memory_id=? AND released=0",
                (memory_id,),
            ).fetchone()
            is not None
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def quarantine(
        self,
        memory_id: str,
        reason: str,
        score: Optional[float] = None,
    ) -> str:
        """Add a memory to quarantine. Returns the quarantine record ID."""
        with self._conn() as conn:
            qid = self._insert(conn, memory_id, reason, score)
        logger.info("[Quarantine] %s quarantined: %s (score=%s)", memory_id, reason, score)
        return qid

    def release(self, memory_id: str) -> bool:
        """Mark a quarantined memory as reviewed and released (safe to restore)."""
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE quarantine SET released=1, reviewed=1 WHERE memory_id=? AND released=0",
                (memory_id,),
            )
            changed = cur.rowcount > 0
        return changed

    def is_quarantined(self, memory_id: str) -> bool:
        with self._conn() as conn:
            return self._held(conn, memory_id)

    def list_quarantined(self, include_expired: bool = False) -> List[Dict]:
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            if include_expired:
                rows = conn.execute(
                    "SELECT id, memory_id, reason, score, quarantined_at, expires_at, reviewed, released "
                    "FROM quarantine WHERE released=0 ORDER BY quarantined_at DESC"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, memory_id, reason, score, quarantined_at, expires_at, reviewed, released "
                    "FROM quarantine WHERE released=0 AND expires_at > ? ORDER BY quarantined_at DESC",
                    (now,),
                ).fetchall()
        cols = [
            "id",
            "memory_id",
            "reason",
            "score",
            "quarantined_at",
            "expires_at",
            "reviewed",
            "released",
        ]
        return [dict(zip(cols, row)) for row in rows]

    def purge_expired(self) -> int:
        """Delete expired quarantine records and their underlying memory rows."""
        now = datetime.now(timezone.utc).isoformat()
        expired_ids: List[str] = []
        # The write lock is taken up front: a release() landing between the
        # SELECT and the DELETE would otherwise have its memory deleted anyway.
        with self._conn(immediate=True) as conn:
            expired_rows = conn.execute(
                "SELECT memory_id FROM quarantine WHERE expires_at <= ? AND released=0 AND reviewed=0",
                (now,),
            ).fetchall()
            expired_ids = [r[0] for r in expired_rows]
            if expired_ids:
                placeholders = ",".join("?" * len(expired_ids))
                conn.execute(
                    f"DELETE FROM quarantine WHERE memory_id IN ({placeholders})",
                    expired_ids,
                )
                try:
                    conn.execute(
                        f"DELETE FROM memories WHERE id IN ({placeholders})",
                        expired_ids,
                    )
                except sqlite3.OperationalError:
                    pass  # memories table may not exist in isolated test DBs
        if expired_ids:
            logger.info("[Quarantine] Purged %d expired memories", len(expired_ids))
        return len(expired_ids)

    def auto_quarantine_by_score(
        self,
        entries: List[Any],
        scores: Dict[str, float],
        threshold: Optional[float] = None,
    ) -> List[str]:
        """
        Quarantine any entries whose composite score is below threshold.
        Skips entries already in quarantine.
        Returns list of memory IDs that were quarantined.
        """
        thr = threshold if threshold is not None else self.LOW_SCORE_THRESHOLD
        quarantined: List[str] = []
        # One transaction for the whole batch: checking "already quarantined" on
        # one connection and inserting on another lets the same memory in twice.
        with self._conn(immediate=True) as conn:
            for entry in entries:
                eid = getattr(entry, "id", None)
                if not eid:
                    continue
                sc = scores.get(eid, 1.0)
                if sc < thr and not self._held(conn, eid):
                    self._insert(conn, eid, f"low_score:{sc:.3f}", sc)
                    quarantined.append(eid)
        if quarantined:
            logger.info("[Quarantine] Auto-quarantined %d low-score memories", len(quarantined))
        return quarantined


_quarantine: Optional[MemoryQuarantine] = None


def get_quarantine() -> MemoryQuarantine:
    global _quarantine
    if _quarantine is None:
        _quarantine = MemoryQuarantine()
    return _quarantine
