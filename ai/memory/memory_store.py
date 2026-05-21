"""
Memory Store for A.L.I.C.E
Abstract storage interface for memory entries
"""

import json
import pickle
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """Single memory entry"""

    id: str
    content: str
    memory_type: str  # "episodic", "semantic", "procedural", "document"
    timestamp: str
    context: Dict[str, Any]
    importance: float = 0.5
    access_count: int = 0
    last_accessed: Optional[str] = None
    embedding: Optional[List[float]] = None
    tags: List[str] = None
    source_file: Optional[str] = None  # Source document path
    chunk_index: Optional[int] = None  # For document chunks

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class MemoryStore(ABC):
    """Abstract interface for memory storage backends"""

    @abstractmethod
    def add(self, entry: MemoryEntry) -> bool:
        """
        Add a single memory entry

        Args:
            entry: Memory entry to add

        Returns:
            True if added successfully
        """
        pass

    @abstractmethod
    def bulk_add(self, entries: List[MemoryEntry]) -> int:
        """
        Add multiple entries efficiently

        Args:
            entries: List of memory entries

        Returns:
            Count of entries added
        """
        pass

    @abstractmethod
    def get_by_id(self, memory_id: str) -> Optional[MemoryEntry]:
        """
        Retrieve memory by ID

        Args:
            memory_id: Memory identifier

        Returns:
            Memory entry or None
        """
        pass

    @abstractmethod
    def get_all(self, memory_type: Optional[str] = None) -> List[MemoryEntry]:
        """
        Get all memories, optionally filtered by type

        Args:
            memory_type: Filter by type (optional)

        Returns:
            List of memory entries
        """
        pass

    @abstractmethod
    def find_by_similarity(
        self,
        embedding: np.ndarray,
        threshold: float,
        top_k: int,
        memory_type: Optional[str] = None,
    ) -> List[MemoryEntry]:
        """
        Find similar memories by vector similarity

        Args:
            embedding: Query embedding vector
            threshold: Minimum similarity threshold
            top_k: Maximum number of results
            memory_type: Filter by type (optional)

        Returns:
            List of similar memories
        """
        pass

    @abstractmethod
    def remove(self, memory_id: str) -> bool:
        """
        Remove memory by ID

        Args:
            memory_id: Memory identifier

        Returns:
            True if removed successfully
        """
        pass

    @abstractmethod
    def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update memory metadata

        Args:
            memory_id: Memory identifier
            updates: Fields to update

        Returns:
            True if updated successfully
        """
        pass

    @abstractmethod
    def count(self, memory_type: Optional[str] = None) -> int:
        """
        Count memories, optionally by type

        Args:
            memory_type: Filter by type (optional)

        Returns:
            Count of memories
        """
        pass


class InMemoryMemoryStore(MemoryStore):
    """In-memory implementation of MemoryStore"""

    def __init__(self) -> None:
        self.memories: Dict[str, MemoryEntry] = {}
        logger.info("[InMemoryMemoryStore] Initialized")

    def add(self, entry: MemoryEntry) -> bool:
        """Add single memory entry"""
        try:
            self.memories[entry.id] = entry
            return True
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            return False

    def bulk_add(self, entries: List[MemoryEntry]) -> int:
        """Add multiple entries"""
        count = 0
        for entry in entries:
            if self.add(entry):
                count += 1
        return count

    def get_by_id(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get memory by ID"""
        return self.memories.get(memory_id)

    def get_all(self, memory_type: Optional[str] = None) -> List[MemoryEntry]:
        """Get all memories"""
        if memory_type is None:
            return list(self.memories.values())

        return [mem for mem in self.memories.values() if mem.memory_type == memory_type]

    def find_by_similarity(
        self,
        embedding: np.ndarray,
        threshold: float,
        top_k: int,
        memory_type: Optional[str] = None,
    ) -> List[MemoryEntry]:
        """Find similar memories"""
        from sklearn.metrics.pairwise import cosine_similarity

        candidates = self.get_all(memory_type=memory_type)

        # Filter memories with embeddings
        memories_with_embeddings = [
            mem for mem in candidates if mem.embedding is not None
        ]

        if not memories_with_embeddings:
            return []

        try:
            # Calculate similarities
            similarities = []
            for mem in memories_with_embeddings:
                mem_embedding = np.array(mem.embedding)
                similarity = cosine_similarity(
                    embedding.reshape(1, -1), mem_embedding.reshape(1, -1)
                )[0][0]

                if similarity >= threshold:
                    similarities.append((mem, similarity))

            # Sort by similarity and take top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return [mem for mem, _ in similarities[:top_k]]

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

    def remove(self, memory_id: str) -> bool:
        """Remove memory by ID"""
        try:
            if memory_id in self.memories:
                del self.memories[memory_id]
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to remove memory: {e}")
            return False

    def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update memory metadata"""
        if memory_id not in self.memories:
            return False

        try:
            memory = self.memories[memory_id]
            for key, value in updates.items():
                if hasattr(memory, key):
                    setattr(memory, key, value)
            return True
        except Exception as e:
            logger.error(f"Failed to update memory: {e}")
            return False

    def count(self, memory_type: Optional[str] = None) -> int:
        """Count memories"""
        if memory_type is None:
            return len(self.memories)

        return sum(
            1 for mem in self.memories.values() if mem.memory_type == memory_type
        )


class SQLiteMemoryStore(MemoryStore):
    """SQLite-backed memory store — ACID writes, WAL reads, embeddings as BLOBs."""

    _DEFAULT_DB = "data/memory/alice.db"

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = Path(db_path or self._DEFAULT_DB)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info(f"[SQLiteMemoryStore] Ready at {self.db_path}")

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id            TEXT PRIMARY KEY,
                    content       TEXT    NOT NULL,
                    memory_type   TEXT    NOT NULL,
                    timestamp     TEXT    NOT NULL,
                    context       TEXT    NOT NULL DEFAULT '{}',
                    importance    REAL             DEFAULT 0.5,
                    access_count  INTEGER          DEFAULT 0,
                    last_accessed TEXT,
                    embedding     BLOB,
                    tags          TEXT             DEFAULT '[]',
                    source_file   TEXT,
                    chunk_index   INTEGER
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_type ON memories(memory_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_ts   ON memories(timestamp DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_imp  ON memories(importance DESC)")
            conn.commit()

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pack(entry: MemoryEntry) -> tuple:
        emb = entry.embedding
        if emb is not None:
            blob = pickle.dumps(np.array(emb, dtype=np.float32))
        else:
            blob = None
        return (
            entry.id,
            entry.content,
            entry.memory_type,
            entry.timestamp,
            json.dumps(entry.context or {}),
            entry.importance,
            entry.access_count,
            entry.last_accessed,
            blob,
            json.dumps(entry.tags or []),
            entry.source_file,
            entry.chunk_index,
        )

    @staticmethod
    def _unpack(row: tuple) -> MemoryEntry:
        (id_, content, memory_type, timestamp, context,
         importance, access_count, last_accessed,
         embedding_blob, tags, source_file, chunk_index) = row
        emb = pickle.loads(embedding_blob).tolist() if embedding_blob else None
        return MemoryEntry(
            id=id_,
            content=content,
            memory_type=memory_type,
            timestamp=timestamp,
            context=json.loads(context) if context else {},
            importance=importance or 0.5,
            access_count=access_count or 0,
            last_accessed=last_accessed,
            embedding=emb,
            tags=json.loads(tags) if tags else [],
            source_file=source_file,
            chunk_index=chunk_index,
        )

    _UPSERT = """
        INSERT OR REPLACE INTO memories
        (id, content, memory_type, timestamp, context, importance,
         access_count, last_accessed, embedding, tags, source_file, chunk_index)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    # ------------------------------------------------------------------
    # MemoryStore interface
    # ------------------------------------------------------------------

    def add(self, entry: MemoryEntry) -> bool:
        try:
            with self._conn() as conn:
                conn.execute(self._UPSERT, self._pack(entry))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"[SQLiteMemoryStore] add failed: {e}")
            return False

    def bulk_add(self, entries: List[MemoryEntry]) -> int:
        if not entries:
            return 0
        try:
            params = [self._pack(e) for e in entries]
            with self._conn() as conn:
                conn.executemany(self._UPSERT, params)
                conn.commit()
            return len(params)
        except Exception as e:
            logger.error(f"[SQLiteMemoryStore] bulk_add failed: {e}")
            return 0

    def get_by_id(self, memory_id: str) -> Optional[MemoryEntry]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        return self._unpack(row) if row else None

    def get_all(self, memory_type: Optional[str] = None) -> List[MemoryEntry]:
        with self._conn() as conn:
            if memory_type:
                rows = conn.execute(
                    "SELECT * FROM memories WHERE memory_type = ? ORDER BY timestamp DESC",
                    (memory_type,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM memories ORDER BY timestamp DESC"
                ).fetchall()
        return [self._unpack(r) for r in rows]

    def find_by_similarity(
        self,
        embedding: np.ndarray,
        threshold: float,
        top_k: int,
        memory_type: Optional[str] = None,
    ) -> List[MemoryEntry]:
        from sklearn.metrics.pairwise import cosine_similarity

        candidates = [m for m in self.get_all(memory_type) if m.embedding is not None]
        if not candidates:
            return []
        try:
            query = np.array(embedding).reshape(1, -1)
            scored = []
            for mem in candidates:
                sim = cosine_similarity(query, np.array(mem.embedding).reshape(1, -1))[0][0]
                if sim >= threshold:
                    scored.append((mem, sim))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [m for m, _ in scored[:top_k]]
        except Exception as e:
            logger.error(f"[SQLiteMemoryStore] similarity search failed: {e}")
            return []

    def remove(self, memory_id: str) -> bool:
        try:
            with self._conn() as conn:
                conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"[SQLiteMemoryStore] remove failed: {e}")
            return False

    def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        _allowed = {"content", "importance", "access_count", "last_accessed", "tags", "context", "embedding"}
        filtered = {k: v for k, v in updates.items() if k in _allowed}
        if not filtered:
            return False
        try:
            clauses, params = [], []
            for k, v in filtered.items():
                clauses.append(f"{k} = ?")
                if k == "tags":
                    params.append(json.dumps(v or []))
                elif k == "context":
                    params.append(json.dumps(v or {}))
                elif k == "embedding":
                    params.append(pickle.dumps(np.array(v, dtype=np.float32)) if v is not None else None)
                else:
                    params.append(v)
            params.append(memory_id)
            with self._conn() as conn:
                conn.execute(f"UPDATE memories SET {', '.join(clauses)} WHERE id = ?", params)
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"[SQLiteMemoryStore] update failed: {e}")
            return False

    def count(self, memory_type: Optional[str] = None) -> int:
        with self._conn() as conn:
            if memory_type:
                return conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE memory_type = ?", (memory_type,)
                ).fetchone()[0]
            return conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def migrate_from_json(self, json_path: str) -> int:
        """One-time migration from memories.json into SQLite."""
        path = Path(json_path)
        if not path.exists():
            return 0
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            buckets = data if isinstance(data, dict) else {"episodic": data}
            entries: List[MemoryEntry] = []
            for bucket_rows in buckets.values():
                for m in bucket_rows:
                    try:
                        entries.append(MemoryEntry(**m))
                    except Exception:
                        pass
            count = self.bulk_add(entries)
            logger.info(f"[SQLiteMemoryStore] Migrated {count} entries from {json_path}")
            return count
        except Exception as e:
            logger.error(f"[SQLiteMemoryStore] Migration failed: {e}")
            return 0


# ------------------------------------------------------------------
# Singleton
# ------------------------------------------------------------------

_memory_store: Optional[SQLiteMemoryStore] = None


def get_memory_store() -> SQLiteMemoryStore:
    """Return the process-wide SQLiteMemoryStore singleton."""
    global _memory_store
    if _memory_store is None:
        _memory_store = SQLiteMemoryStore()
    return _memory_store
