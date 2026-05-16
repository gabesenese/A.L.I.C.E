from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

from ai.memory.alice_memory_schema import (
    ActiveConceptThread,
    MemoryRecord,
    RetrievedMemory,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def _build_fts_match_query(query: str) -> str:
    tokens = _tokenize(query)
    if not tokens:
        return ""
    return " AND ".join(f"\"{token}\"" for token in tokens)


class AliceMemoryService:
    def __init__(self, db_path: str = "data/alice_memory/memory.db") -> None:
        self.db_path = Path(db_path)
        self._fts_enabled = False

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    topic TEXT NOT NULL DEFAULT '',
                    confidence REAL NOT NULL DEFAULT 0.5,
                    importance INTEGER NOT NULL DEFAULT 5,
                    source TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_topic ON memories(topic)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at)")
            try:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
                    USING fts5(memory_id UNINDEXED, content, topic)
                    """
                )
                self._fts_enabled = True
            except sqlite3.OperationalError:
                self._fts_enabled = False
            if self._fts_enabled:
                self._rebuild_fts_index(conn)
            conn.commit()

    def _rebuild_fts_index(self, conn: sqlite3.Connection) -> None:
        if not self._fts_enabled:
            return
        conn.execute("DELETE FROM memory_fts")
        rows = conn.execute("SELECT id, content, topic FROM memories").fetchall()
        for row in rows:
            conn.execute(
                "INSERT INTO memory_fts(memory_id, content, topic) VALUES(?, ?, ?)",
                (str(row["id"]), str(row["content"] or ""), str(row["topic"] or "")),
            )

    @staticmethod
    def _record_from_row(row: sqlite3.Row) -> MemoryRecord:
        metadata_json = str(row["metadata_json"] or "{}")
        try:
            metadata = dict(json.loads(metadata_json))
        except Exception:
            metadata = {}
        return MemoryRecord(
            id=str(row["id"] or ""),
            kind=str(row["kind"] or ""),
            content=str(row["content"] or ""),
            topic=str(row["topic"] or ""),
            confidence=float(row["confidence"] or 0.0),
            importance=int(row["importance"] or 0),
            source=str(row["source"] or ""),
            created_at=str(row["created_at"] or ""),
            updated_at=str(row["updated_at"] or ""),
            metadata=metadata,
        )

    def save_memory(self, record: MemoryRecord) -> MemoryRecord:
        now = _utc_now_iso()
        memory_id = str(record.id or f"mem_{uuid4().hex[:14]}")
        created_at = str(record.created_at or now)
        updated_at = str(now)
        metadata_json = json.dumps(dict(record.metadata or {}), ensure_ascii=False)
        saved = MemoryRecord(
            id=memory_id,
            kind=str(record.kind or "interaction"),
            content=str(record.content or ""),
            topic=str(record.topic or ""),
            confidence=float(record.confidence or 0.0),
            importance=int(record.importance or 0),
            source=str(record.source or ""),
            created_at=created_at,
            updated_at=updated_at,
            metadata=dict(record.metadata or {}),
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memories(
                    id, kind, content, topic, confidence, importance, source,
                    created_at, updated_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    saved.id,
                    saved.kind,
                    saved.content,
                    saved.topic,
                    saved.confidence,
                    saved.importance,
                    saved.source,
                    saved.created_at,
                    saved.updated_at,
                    metadata_json,
                ),
            )
            if self._fts_enabled:
                conn.execute("DELETE FROM memory_fts WHERE memory_id = ?", (saved.id,))
                conn.execute(
                    "INSERT INTO memory_fts(memory_id, content, topic) VALUES (?, ?, ?)",
                    (saved.id, saved.content, saved.topic),
                )
            conn.commit()
        return saved

    def save_fact(
        self,
        content: str,
        topic: str = "",
        confidence: float = 0.7,
        importance: int = 5,
        source: str = "",
    ) -> MemoryRecord:
        return self.save_memory(
            MemoryRecord(
                id=f"mem_{uuid4().hex[:14]}",
                kind="fact",
                content=str(content or ""),
                topic=str(topic or ""),
                confidence=float(confidence),
                importance=int(importance),
                source=str(source or ""),
                created_at=_utc_now_iso(),
                metadata={},
            )
        )

    def save_concept_thread(self, thread: ActiveConceptThread) -> MemoryRecord:
        payload = asdict(thread)
        return self.save_memory(
            MemoryRecord(
                id=f"mem_{uuid4().hex[:14]}",
                kind="concept_thread",
                content=str(thread.topic or "proactive AI companion"),
                topic=str(thread.topic or "proactive AI companion"),
                confidence=float(thread.confidence or 0.8),
                importance=9,
                source="alice_companion_core",
                created_at=_utc_now_iso(),
                metadata={"thread": payload},
            )
        )

    def get_active_concept_thread(self) -> Optional[ActiveConceptThread]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM memories
                WHERE kind = 'concept_thread'
                ORDER BY created_at DESC
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            return None
        record = self._record_from_row(row)
        payload = dict(record.metadata.get("thread") or {})
        if payload:
            try:
                return ActiveConceptThread(
                    topic=str(payload.get("topic") or record.topic or "proactive AI companion"),
                    constraints=list(payload.get("constraints") or []),
                    signals=list(payload.get("signals") or []),
                    last_user_inputs=list(payload.get("last_user_inputs") or []),
                    updated_at=str(payload.get("updated_at") or record.updated_at or record.created_at),
                    confidence=float(payload.get("confidence") or record.confidence or 0.8),
                )
            except Exception:
                pass
        return ActiveConceptThread(
            topic=str(record.topic or "proactive AI companion"),
            constraints=[],
            signals=[],
            last_user_inputs=[],
            updated_at=str(record.updated_at or record.created_at or _utc_now_iso()),
            confidence=float(record.confidence or 0.8),
        )

    @staticmethod
    def _concept_signal_map() -> Dict[str, str]:
        return {
            "ai companion": "ai companion",
            "agentic ai": "agentic ai",
            "proactive companion": "proactive companion",
            "not assistant": "not assistant",
            "not chatbot": "not chatbot",
            "always running": "always running",
            "background monitoring": "background monitoring",
            "detect changes": "detect changes",
            "suggest actions": "suggest actions",
            "local-first": "local-first",
            "alice-ollama": "alice-ollama",
            "alice ollama": "alice-ollama",
            "brain": "brain",
        }

    @staticmethod
    def _constraint_map() -> Dict[str, str]:
        return {
            "not assistant": "not generic assistant",
            "not chatbot": "not chatbot",
            "proactive": "proactive",
            "always running": "always-running",
            "background monitoring": "background monitoring",
            "detect changes": "detects changes",
            "suggest actions": "suggests actions",
            "local-first": "local-first",
            "alice-ollama": "alice-ollama as brain",
            "alice ollama": "alice-ollama as brain",
        }

    def update_active_concept_thread(
        self, user_input: str, topic_hint: str = ""
    ) -> Optional[ActiveConceptThread]:
        text = str(user_input or "").lower()
        existing = self.get_active_concept_thread()
        signal_map = self._concept_signal_map()
        constraint_map = self._constraint_map()

        detected_signals: List[str] = []
        for marker, signal in signal_map.items():
            if marker in text and signal not in detected_signals:
                detected_signals.append(signal)

        has_proactive_signal = any(
            marker in text
            for marker in (
                "proactive",
                "always running",
                "background monitoring",
                "detect changes",
                "suggest actions",
                "ai companion",
                "agentic ai",
            )
        )
        if not has_proactive_signal and existing is None:
            return None

        topic = str(topic_hint or (existing.topic if existing else "") or "proactive AI companion")
        constraints = list(existing.constraints if existing else [])
        signals = list(existing.signals if existing else [])
        if "approval before risky actions" not in constraints:
            constraints.append("approval before risky actions")

        if has_proactive_signal and "proactive" not in constraints:
            constraints.append("proactive")

        for marker, label in constraint_map.items():
            if marker in text and label not in constraints:
                constraints.append(label)
        for signal in detected_signals:
            if signal not in signals:
                signals.append(signal)

        last_inputs = list(existing.last_user_inputs if existing else [])
        user_line = str(user_input or "").strip()
        if user_line:
            last_inputs.append(user_line)
            last_inputs = last_inputs[-6:]

        thread = ActiveConceptThread(
            topic=topic,
            constraints=constraints,
            signals=signals,
            last_user_inputs=last_inputs,
            updated_at=_utc_now_iso(),
            confidence=max(0.8, float(existing.confidence if existing else 0.8)),
        )
        self.save_concept_thread(thread)
        return thread

    @staticmethod
    def _recency_score(created_at: str) -> float:
        try:
            created = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
            delta_hours = max(
                0.0,
                (datetime.now(timezone.utc) - created.astimezone(timezone.utc)).total_seconds() / 3600.0,
            )
            return max(0.0, 1.0 - (delta_hours / (24.0 * 21.0)))
        except Exception:
            return 0.0

    @staticmethod
    def _overlap_ratio(query: str, content: str, topic: str) -> float:
        q = set(_tokenize(query))
        if not q:
            return 0.0
        c = set(_tokenize(content) + _tokenize(topic))
        if not c:
            return 0.0
        return float(len(q.intersection(c))) / float(len(q))

    def _score_record(
        self, record: MemoryRecord, query: str, *, matched_by_fts: bool
    ) -> float:
        overlap = self._overlap_ratio(query, record.content, record.topic)
        recency = self._recency_score(record.created_at)
        confidence = max(0.0, min(1.0, float(record.confidence or 0.0)))
        importance = max(0, min(10, int(record.importance or 0))) / 10.0
        match_bonus = 1.0 if matched_by_fts else overlap
        score = (
            (match_bonus * 0.55)
            + (overlap * 0.2)
            + (confidence * 0.15)
            + (importance * 0.07)
            + (recency * 0.03)
        )
        return round(score, 6)

    def _search_with_fts(self, query: str, kind: Optional[str], limit: int) -> List[MemoryRecord]:
        if not self._fts_enabled:
            return []
        fts_query = _build_fts_match_query(query)
        if not fts_query:
            return []
        with self._connect() as conn:
            if kind:
                rows = conn.execute(
                    """
                    SELECT m.* FROM memory_fts f
                    JOIN memories m ON m.id = f.memory_id
                    WHERE memory_fts MATCH ? AND m.kind = ?
                    LIMIT ?
                    """,
                    (fts_query, str(kind), int(limit)),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT m.* FROM memory_fts f
                    JOIN memories m ON m.id = f.memory_id
                    WHERE memory_fts MATCH ?
                    LIMIT ?
                    """,
                    (fts_query, int(limit)),
                ).fetchall()
        return [self._record_from_row(row) for row in rows]

    def _search_with_keywords(self, query: str, kind: Optional[str], limit: int) -> List[MemoryRecord]:
        with self._connect() as conn:
            if kind:
                rows = conn.execute(
                    """
                    SELECT * FROM memories
                    WHERE kind = ?
                    ORDER BY created_at DESC
                    LIMIT 250
                    """,
                    (str(kind),),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM memories
                    ORDER BY created_at DESC
                    LIMIT 250
                    """
                ).fetchall()
        records = [self._record_from_row(row) for row in rows]
        scored = sorted(
            records,
            key=lambda rec: self._overlap_ratio(query, rec.content, rec.topic),
            reverse=True,
        )
        return [rec for rec in scored if self._overlap_ratio(query, rec.content, rec.topic) > 0.0][:limit]

    def search_memories(
        self, query: str, *, kind: Optional[str] = None, limit: int = 5
    ) -> List[RetrievedMemory]:
        q = str(query or "").strip()
        if not q:
            return []
        safe_limit = max(1, int(limit or 1))
        candidates = self._search_with_fts(q, kind, safe_limit * 3)
        matched_by_fts_ids = {rec.id for rec in candidates}
        if not candidates:
            candidates = self._search_with_keywords(q, kind, safe_limit * 3)

        out: List[RetrievedMemory] = []
        for record in candidates:
            matched_by_fts = record.id in matched_by_fts_ids
            score = self._score_record(record, q, matched_by_fts=matched_by_fts)
            if score <= 0.0:
                continue
            confidence_label = "verified" if float(record.confidence or 0.0) >= 0.65 else "hint"
            reason = "fts_match" if matched_by_fts else "keyword_overlap"
            out.append(
                RetrievedMemory(
                    record=record,
                    score=score,
                    reason=reason,
                    confidence_label=confidence_label,
                )
            )
        out.sort(key=lambda item: item.score, reverse=True)
        return out[:safe_limit]

    def get_recent_memories(self, limit: int = 10) -> List[MemoryRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM memories
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (max(1, int(limit or 1)),),
            ).fetchall()
        return [self._record_from_row(row) for row in rows]
