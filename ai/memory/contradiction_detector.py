from __future__ import annotations

import logging
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_DB_PATH = Path("data/memory/alice.db")

_NEGATION_RE = re.compile(
    r"\b(not|never|no|isn'?t|aren'?t|wasn'?t|weren'?t|doesn'?t|don'?t|didn'?t|"
    r"won'?t|wouldn'?t|can'?t|couldn'?t|shouldn'?t|no longer|false|incorrect|wrong)\b",
    re.IGNORECASE,
)

# A pair is a contradiction candidate when similarity is in this range:
# too-low → unrelated; too-high → exact duplicate (handled by dedup)
_SIM_LOW = 0.60
_SIM_HIGH = 0.95
_MIN_CONFIDENCE = 0.45


class ContradictionDetector:
    """
    Detects semantic contradictions between memory pairs.

    A contradiction is detected when two memories:
    1. Are about the same topic   (embedding cosine similarity 0.60–0.95)
    2. Show negation asymmetry    (one negates something the other asserts)

    Detected pairs are stored in the `contradictions` table for human review.
    The detector does NOT auto-delete either memory; resolution is manual.

    Schema: contradictions table in alice.db
    """

    def __init__(self, db_path: Path = _DB_PATH) -> None:
        self.db_path = db_path
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS contradictions (
                    id           TEXT PRIMARY KEY,
                    memory_a_id  TEXT NOT NULL,
                    memory_b_id  TEXT NOT NULL,
                    confidence   REAL NOT NULL,
                    detected_at  TEXT NOT NULL,
                    resolved     INTEGER DEFAULT 0,
                    resolution   TEXT,
                    UNIQUE(memory_a_id, memory_b_id)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_contr_a ON contradictions(memory_a_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_contr_b ON contradictions(memory_b_id)")
            conn.commit()

    # ------------------------------------------------------------------
    # Detection logic
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))

    @staticmethod
    def _neg_count(text: str) -> int:
        return len(_NEGATION_RE.findall(text))

    def _confidence(self, content_a: str, content_b: str, sim: float) -> float:
        """
        Estimate contradiction confidence from negation asymmetry + topic similarity.

        High topic similarity + strong negation asymmetry → high confidence.
        """
        diff = abs(self._neg_count(content_a) - self._neg_count(content_b))
        asymmetry = min(1.0, diff * 0.30)
        # Normalise similarity into the candidate window
        topic_weight = (sim - _SIM_LOW) / (_SIM_HIGH - _SIM_LOW)
        return min(1.0, asymmetry * topic_weight)

    def check_pair(self, entry_a: Any, entry_b: Any) -> Optional[float]:
        """
        Check whether two entries contradict each other.
        Returns confidence [0,1] or None if no contradiction.
        """
        emb_a = getattr(entry_a, "embedding", None)
        emb_b = getattr(entry_b, "embedding", None)
        if not emb_a or not emb_b:
            return None

        sim = self._cosine(emb_a, emb_b)
        if sim < _SIM_LOW or sim >= _SIM_HIGH:
            return None

        content_a = str(getattr(entry_a, "content", "") or "")
        content_b = str(getattr(entry_b, "content", "") or "")
        conf = self._confidence(content_a, content_b, sim)
        return conf if conf >= _MIN_CONFIDENCE else None

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def record(self, id_a: str, id_b: str, confidence: float) -> str:
        """Persist a contradiction pair. Canonical order prevents (a,b)/(b,a) dupes."""
        if id_a > id_b:
            id_a, id_b = id_b, id_a
        cid = f"c_{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO contradictions
                    (id, memory_a_id, memory_b_id, confidence, detected_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (cid, id_a, id_b, confidence, now),
            )
            conn.commit()
        logger.info("[ContradictionDetector] %s ↔ %s (conf=%.2f)", id_a, id_b, confidence)
        return cid

    def resolve(self, memory_a_id: str, memory_b_id: str, resolution: str) -> bool:
        if memory_a_id > memory_b_id:
            memory_a_id, memory_b_id = memory_b_id, memory_a_id
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE contradictions SET resolved=1, resolution=? WHERE memory_a_id=? AND memory_b_id=?",
                (resolution, memory_a_id, memory_b_id),
            )
            conn.commit()
        return cur.rowcount > 0

    def list_unresolved(self) -> List[Dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, memory_a_id, memory_b_id, confidence, detected_at "
                "FROM contradictions WHERE resolved=0 ORDER BY confidence DESC"
            ).fetchall()
        return [
            {
                "id": r[0],
                "memory_a_id": r[1],
                "memory_b_id": r[2],
                "confidence": r[3],
                "detected_at": r[4],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Batch scan
    # ------------------------------------------------------------------

    def scan(
        self,
        entries: List[Any],
        *,
        max_pairs: int = 500,
    ) -> List[Tuple[str, str, float]]:
        """
        Scan a list of entries for contradictions (O(n²) capped at max_pairs).
        Persists each detected pair and returns list of (id_a, id_b, confidence).
        """
        found: List[Tuple[str, str, float]] = []
        checked = 0
        for i, ea in enumerate(entries):
            if checked >= max_pairs:
                break
            for j in range(i + 1, len(entries)):
                if checked >= max_pairs:
                    break
                checked += 1
                conf = self.check_pair(ea, entries[j])
                if conf is not None:
                    id_a = getattr(ea, "id", str(i))
                    id_b = getattr(entries[j], "id", str(j))
                    self.record(id_a, id_b, conf)
                    found.append((id_a, id_b, conf))
        return found


_detector: Optional[ContradictionDetector] = None


def get_contradiction_detector() -> ContradictionDetector:
    global _detector
    if _detector is None:
        _detector = ContradictionDetector()
    return _detector
