from __future__ import annotations

import logging
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DB_PATH = Path("data/memory/alice.db")

# Patterns signalling a causal relationship within a sentence
_CAUSAL_MARKER_RE = re.compile(
    r"\b(because|since|due to|as a result of|caused by|led to|triggered|"
    r"resulted in|therefore|consequently|thus|hence|so that|in order to|"
    r"which means|which caused|after which)\b",
    re.IGNORECASE,
)

# Used to split a sentence at the causal pivot
_SPLIT_MARKER_RE = re.compile(
    r"\b(because|since|therefore|consequently|thus|hence|led to|"
    r"resulted in|caused|triggered|so that)\b",
    re.IGNORECASE,
)

# Markers where the clause order is effect–marker–cause
_EFFECT_FIRST_MARKERS = {"because", "since"}


class CausalMemory:
    """
    Stores and retrieves causal chains between memories.

    A causal chain: cause_memory → effect_memory

    Chains are extracted when conversation contains causative language, or
    recorded explicitly when two sequential memories are likely related.

    Schema: causal_chains table in alice.db
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
                CREATE TABLE IF NOT EXISTS causal_chains (
                    id          TEXT PRIMARY KEY,
                    cause_id    TEXT NOT NULL,
                    effect_id   TEXT NOT NULL,
                    chain_type  TEXT NOT NULL DEFAULT 'explicit',
                    confidence  REAL NOT NULL DEFAULT 1.0,
                    description TEXT,
                    detected_at TEXT NOT NULL,
                    UNIQUE(cause_id, effect_id)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_causal_cause  ON causal_chains(cause_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_causal_effect ON causal_chains(effect_id)"
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def record(
        self,
        cause_id: str,
        effect_id: str,
        *,
        chain_type: str = "explicit",
        confidence: float = 1.0,
        description: str = "",
    ) -> str:
        """Persist a causal link. Returns chain ID."""
        cid = f"chain_{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO causal_chains
                    (id, cause_id, effect_id, chain_type, confidence, description, detected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (cid, cause_id, effect_id, chain_type, round(confidence, 4), description, now),
            )
            conn.commit()
        logger.debug(
            "[CausalMemory] %s → %s (%s, conf=%.2f)", cause_id, effect_id, chain_type, confidence
        )
        return cid

    def link_sequential(
        self,
        earlier_id: str,
        later_id: str,
        confidence: float = 0.40,
    ) -> str:
        """Record an inferred causal link based on temporal proximity."""
        return self.record(
            earlier_id, later_id,
            chain_type="inferred",
            confidence=confidence,
            description="temporal_proximity",
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_effects(self, cause_id: str) -> List[Dict]:
        """Return all effect entries caused by this memory."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT effect_id, chain_type, confidence, description "
                "FROM causal_chains WHERE cause_id=? ORDER BY confidence DESC",
                (cause_id,),
            ).fetchall()
        return [
            {"effect_id": r[0], "chain_type": r[1], "confidence": r[2], "description": r[3]}
            for r in rows
        ]

    def get_causes(self, effect_id: str) -> List[Dict]:
        """Return all causes that produced this memory."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT cause_id, chain_type, confidence, description "
                "FROM causal_chains WHERE effect_id=? ORDER BY confidence DESC",
                (effect_id,),
            ).fetchall()
        return [
            {"cause_id": r[0], "chain_type": r[1], "confidence": r[2], "description": r[3]}
            for r in rows
        ]

    def get_chain_context(self, memory_id: str) -> str:
        """One-line causal context for a memory (causes + effects), for LLM injection."""
        causes  = self.get_causes(memory_id)
        effects = self.get_effects(memory_id)
        parts: List[str] = []
        if causes:
            parts.append("caused by: " + ", ".join(c["cause_id"] for c in causes))
        if effects:
            parts.append("led to: " + ", ".join(e["effect_id"] for e in effects))
        return "; ".join(parts)

    # ------------------------------------------------------------------
    # Text extraction
    # ------------------------------------------------------------------

    def extract_from_text(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract (cause_text, effect_text) pairs from free text using causal markers.
        These are text snippets, not memory IDs — callers must store them.
        """
        if not _CAUSAL_MARKER_RE.search(text):
            return []

        pairs: List[Tuple[str, str]] = []
        sentences = re.split(r"[.!?]\s+", text)

        for sent in sentences:
            m = _SPLIT_MARKER_RE.search(sent)
            if not m:
                continue
            marker = m.group(0).lower()
            pivot  = m.start()
            after  = sent[pivot + len(m.group(0)):].strip()
            before = sent[:pivot].strip()

            if marker in _EFFECT_FIRST_MARKERS:
                # "effect because cause"
                cause_text, effect_text = after, before
            else:
                # "cause therefore effect"
                cause_text, effect_text = before, after

            if len(cause_text) > 10 and len(effect_text) > 10:
                pairs.append((cause_text, effect_text))

        return pairs


_causal_memory: Optional[CausalMemory] = None


def get_causal_memory() -> CausalMemory:
    global _causal_memory
    if _causal_memory is None:
        _causal_memory = CausalMemory()
    return _causal_memory
