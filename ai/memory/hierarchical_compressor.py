from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from ai.memory.memory_store import ensure_memory_schema, sqlite_connection

logger = logging.getLogger(__name__)

_DB_PATH = Path("data/memory/alice.db")

LEVEL_RAW = 0  # Individual episodic memories
LEVEL_DAY = 1  # Day summaries  (compressed from raw)
LEVEL_WEEK = 2  # Week summaries (compressed from day)
LEVEL_TOPIC = 3  # Topic summaries (compressed from week)

# Compress when count at this level exceeds the threshold
_THRESHOLDS = {
    LEVEL_RAW: 200,
    LEVEL_DAY: 30,
    LEVEL_WEEK: 8,
}


class HierarchicalCompressor:
    """
    Hierarchical memory compression.

    Raw episodic memories → Day summaries → Week summaries → Topic summaries

    When a level's count exceeds its threshold a compression pass runs:
    - Groups entries by time bucket (day / week)
    - Writes a summary memory (type=semantic, memory_level=N+1) to the DB
    - Links source entries to the summary via parent_id

    The source entries are NOT deleted — they stay in the DB with their
    parent_id set so the hierarchy can be walked or rebuilt.

    The columns this class needs (memories.memory_level, memories.parent_id)
    are declared alongside the rest of the table in ai.memory.memory_store, so
    the reader and the writer of the schema cannot disagree about it.
    """

    def __init__(self, db_path: Path = _DB_PATH) -> None:
        self.db_path = db_path
        self._init_schema()

    def _init_schema(self) -> None:
        with sqlite_connection(self.db_path) as conn:
            ensure_memory_schema(conn)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fetch_level(conn: sqlite3.Connection, level: int) -> List[Dict]:
        """Return all episodic memories at the given compression level."""
        rows = conn.execute(
            """
            SELECT id, content, timestamp, tags, importance
            FROM memories
            WHERE memory_type='episodic'
              AND (memory_level IS NULL OR memory_level=?)
              AND (parent_id IS NULL OR parent_id='')
            ORDER BY timestamp ASC
            """,
            (level,),
        ).fetchall()
        return [
            {
                "id": r[0],
                "content": r[1],
                "timestamp": r[2],
                "tags": json.loads(r[3] or "[]"),
                "importance": float(r[4] or 0.5),
            }
            for r in rows
        ]

    @staticmethod
    def _day_key(ts: str) -> str:
        try:
            return ts[:10]
        except (TypeError, IndexError):
            return "unknown"

    @staticmethod
    def _week_key(ts: str) -> str:
        try:
            d = date.fromisoformat(ts[:10])
            yr, wk, _ = d.isocalendar()
            return f"{yr}-W{wk:02d}"
        except Exception:
            return "unknown"

    def _summarise(self, entries: List[Dict], label: str) -> str:
        """Build a plain-text summary from a group of entries."""
        unique = list(dict.fromkeys(e["content"] for e in entries if e.get("content")))[:10]
        snippets = " | ".join(c[:80] for c in unique)
        return f"[{label}] {len(entries)} memories: {snippets}"

    @staticmethod
    def _write_summary(
        conn: sqlite3.Connection,
        content: str,
        level: int,
        source_ids: List[str],
        importance: float,
    ) -> str:
        mid = f"summary_{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            INSERT OR REPLACE INTO memories
                (id, content, memory_type, timestamp, context,
                 importance, tags, memory_level)
            VALUES (?, ?, 'semantic', ?, ?, ?, '["summary"]', ?)
            """,
            (
                mid,
                content,
                now,
                json.dumps({"compressed_from": source_ids, "compression_level": level}),
                round(importance, 4),
                level,
            ),
        )
        if source_ids:
            placeholders = ",".join("?" * len(source_ids))
            # Never re-parent a row another summary already claimed, so an entry
            # belongs to exactly one summary however two passes interleave.
            conn.execute(
                f"UPDATE memories SET parent_id=? WHERE id IN ({placeholders}) AND (parent_id IS NULL OR parent_id='')",
                [mid] + source_ids,
            )
        return mid

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress_level(self, level: int) -> int:
        """
        Run one compression pass for the given level.
        Returns the number of summary memories created.
        """
        # Selecting the unparented entries and claiming them has to be one
        # transaction: two passes reading the same rows would each write their
        # own summary of them and double-count the history.
        with sqlite_connection(self.db_path, immediate=True) as conn:
            entries = self._fetch_level(conn, level)
            threshold = _THRESHOLDS.get(level, 9_999)
            if len(entries) <= threshold:
                return 0

            created = 0

            if level == LEVEL_RAW:
                groups: Dict[str, List[Dict]] = {}
                for e in entries:
                    groups.setdefault(self._day_key(e["timestamp"]), []).append(e)
                for day, group in groups.items():
                    if len(group) < 3:
                        continue
                    avg_imp = sum(e["importance"] for e in group) / len(group)
                    summary = self._summarise(group, day)
                    self._write_summary(conn, summary, LEVEL_DAY, [e["id"] for e in group], avg_imp)
                    created += 1

            elif level == LEVEL_DAY:
                groups = {}
                for e in entries:
                    groups.setdefault(self._week_key(e["timestamp"]), []).append(e)
                for week, group in groups.items():
                    if len(group) < 3:
                        continue
                    avg_imp = sum(e["importance"] for e in group) / len(group)
                    summary = self._summarise(group, f"Week {week}")
                    self._write_summary(conn, summary, LEVEL_WEEK, [e["id"] for e in group], avg_imp)
                    created += 1

            elif level == LEVEL_WEEK:
                avg_imp = sum(e["importance"] for e in entries) / len(entries)
                summary = self._summarise(entries, "Long-term")
                self._write_summary(conn, summary, LEVEL_TOPIC, [e["id"] for e in entries], avg_imp)
                created = 1

        if created:
            logger.info(
                "[HierarchicalCompressor] Level %d: created %d summaries from %d entries",
                level,
                created,
                len(entries),
            )
        return created

    def run_full_pass(self) -> Dict[int, int]:
        """Run compression for all levels bottom-up. Returns {level: summaries_created}."""
        result: Dict[int, int] = {}
        for level in [LEVEL_RAW, LEVEL_DAY, LEVEL_WEEK]:
            n = self.compress_level(level)
            if n:
                result[level] = n
        return result


_compressor: Optional[HierarchicalCompressor] = None


def get_compressor() -> HierarchicalCompressor:
    global _compressor
    if _compressor is None:
        _compressor = HierarchicalCompressor()
    return _compressor
