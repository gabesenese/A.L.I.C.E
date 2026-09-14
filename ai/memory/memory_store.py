"""
Memory Store for A.L.I.C.E
Abstract storage interface for memory entries
"""

import json
import pickle
import os
import sqlite3
from abc import ABC, abstractmethod
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import ContextManager, Iterator, List, Dict, Optional, Any, Sequence, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Canonical `memories` schema
#
# Every module that touches this table reads these declarations instead of
# writing its own DDL, and every read names MEMORY_COLUMNS instead of using
# SELECT *. A bare SELECT * plus a positional unpack means the day another
# module adds a column, every read raises and the assistant silently comes up
# with an empty memory.
# ------------------------------------------------------------------

_MEMORY_COLUMN_DEFS: Tuple[Tuple[str, str], ...] = (
    ("id", "TEXT PRIMARY KEY"),
    ("content", "TEXT NOT NULL"),
    ("memory_type", "TEXT NOT NULL"),
    ("timestamp", "TEXT NOT NULL"),
    ("context", "TEXT NOT NULL DEFAULT '{}'"),
    ("importance", "REAL DEFAULT 0.5"),
    ("access_count", "INTEGER DEFAULT 0"),
    ("last_accessed", "TEXT"),
    ("embedding", "BLOB"),
    ("tags", "TEXT DEFAULT '[]'"),
    ("source_file", "TEXT"),
    ("chunk_index", "INTEGER"),
)

# Columns owned by ai.memory.hierarchical_compressor. They are declared here so
# a fresh database and one upgraded by ALTER TABLE end up identical.
MEMORY_EXTENSION_COLUMN_DEFS: Tuple[Tuple[str, str], ...] = (
    ("memory_level", "INTEGER DEFAULT 0"),
    ("parent_id", "TEXT"),
)

MEMORY_COLUMNS: Tuple[str, ...] = tuple(name for name, _ in _MEMORY_COLUMN_DEFS)
_MEMORY_SELECT = ", ".join(MEMORY_COLUMNS)

_MEMORY_INDEXES: Tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_mem_type   ON memories(memory_type)",
    "CREATE INDEX IF NOT EXISTS idx_mem_ts     ON memories(timestamp DESC)",
    "CREATE INDEX IF NOT EXISTS idx_mem_imp    ON memories(importance DESC)",
    "CREATE INDEX IF NOT EXISTS idx_mem_level  ON memories(memory_level)",
    "CREATE INDEX IF NOT EXISTS idx_mem_parent ON memories(parent_id)",
    "CREATE INDEX IF NOT EXISTS idx_mem_emb    ON memories(memory_type) WHERE embedding IS NOT NULL",
)


def ensure_memory_schema(conn: sqlite3.Connection) -> None:
    """Create or upgrade the `memories` table in place. Idempotent."""
    columns = ", ".join(f"{name} {defn}" for name, defn in _MEMORY_COLUMN_DEFS)
    conn.execute(f"CREATE TABLE IF NOT EXISTS memories ({columns})")

    present = {row[1] for row in conn.execute("PRAGMA table_info(memories)")}
    for name, defn in MEMORY_EXTENSION_COLUMN_DEFS:
        if name not in present:
            conn.execute(f"ALTER TABLE memories ADD COLUMN {name} {defn}")

    for statement in _MEMORY_INDEXES:
        conn.execute(statement)


@contextmanager
def sqlite_connection(
    db_path: Any,
    *,
    timeout: float = 10.0,
    immediate: bool = False,
) -> Iterator[sqlite3.Connection]:
    """Open the memory database, commit on success, roll back on error, always close.

    sqlite3's own connection context manager commits but never closes, so
    `with sqlite3.connect(...) as conn` leaks a file descriptor per call — a few
    hours of recalls is enough to hit the process limit.

    immediate=True takes the write lock before the first read, which is what a
    read-then-write sequence needs to stay atomic against a concurrent writer.
    """
    conn = sqlite3.connect(str(db_path), timeout=timeout, check_same_thread=False)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        if immediate:
            conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.commit()
    except BaseException:
        with suppress(sqlite3.Error):
            conn.rollback()
        raise
    finally:
        conn.close()


def _rank_by_cosine(
    query: Any,
    vectors: Sequence[np.ndarray],
    threshold: float,
    top_k: int,
) -> List[int]:
    """Indices of `vectors` scoring >= threshold against `query`, best first, capped at top_k.

    One matrix operation over the stacked candidates: scoring row by row made the
    cost of a single recall grow with the size of the whole table.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    if top_k <= 0 or not len(vectors):
        return []

    q = np.asarray(query, dtype=np.float32).reshape(1, -1)
    dim = q.shape[1]
    # Embeddings from a previous model have a different width and cannot be
    # compared; dropping them beats letting vstack fail the whole recall.
    usable = [(i, v) for i, v in enumerate(vectors) if v is not None and v.shape == (dim,)]
    if not usable:
        return []

    origin = np.array([i for i, _ in usable], dtype=np.int64)
    matrix = np.vstack([v for _, v in usable])
    sims = cosine_similarity(q, matrix)[0]

    keep = np.flatnonzero(sims >= threshold)
    if keep.size == 0:
        return []
    order = keep[np.argsort(-sims[keep], kind="stable")][:top_k]
    return [int(origin[i]) for i in order]


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
        candidates = [mem for mem in self.get_all(memory_type=memory_type) if mem.embedding is not None]
        if not candidates:
            return []

        try:
            vectors = [np.asarray(mem.embedding, dtype=np.float32).ravel() for mem in candidates]
            return [candidates[i] for i in _rank_by_cosine(embedding, vectors, threshold, top_k)]
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

        return sum(1 for mem in self.memories.values() if mem.memory_type == memory_type)


class SQLiteMemoryStore(MemoryStore):
    """SQLite-backed memory store — ACID writes, WAL reads, embeddings as BLOBs."""

    _DEFAULT_DB = "data/memory/alice.db"

    def __init__(self, db_path: Optional[str] = None) -> None:
        # ALICE_MEMORY_DB exists because the path was a bare constant, so every
        # test that touched the memory system wrote to the user's live database.
        # That is data loss waiting to happen on its own, and running several
        # processes against one SQLite file is also how it ends up reporting
        # "database disk image is malformed" — after which the session runs with
        # no recall at all.
        self.db_path = Path(db_path or os.getenv("ALICE_MEMORY_DB") or self._DEFAULT_DB)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info(f"[SQLiteMemoryStore] Ready at {self.db_path}")

    def _conn(self) -> ContextManager[sqlite3.Connection]:
        return sqlite_connection(self.db_path)

    def _init_db(self) -> None:
        with self._conn() as conn:
            ensure_memory_schema(conn)

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
    def _unpack(row: Sequence[Any]) -> MemoryEntry:
        # Tolerate a wider row than MEMORY_COLUMNS so a stray SELECT * against a
        # table another module has extended still yields an entry rather than
        # blowing up the caller's whole load.
        (
            id_,
            content,
            memory_type,
            timestamp,
            context,
            importance,
            access_count,
            last_accessed,
            embedding_blob,
            tags,
            source_file,
            chunk_index,
        ) = tuple(row)[: len(MEMORY_COLUMNS)]
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

    _UPSERT = f"""
        INSERT OR REPLACE INTO memories ({_MEMORY_SELECT})
        VALUES ({", ".join("?" * len(MEMORY_COLUMNS))})
    """

    # ------------------------------------------------------------------
    # MemoryStore interface
    # ------------------------------------------------------------------

    def add(self, entry: MemoryEntry) -> bool:
        try:
            with self._conn() as conn:
                conn.execute(self._UPSERT, self._pack(entry))
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
            return len(params)
        except Exception as e:
            logger.error(f"[SQLiteMemoryStore] bulk_add failed: {e}")
            return 0

    def get_by_id(self, memory_id: str) -> Optional[MemoryEntry]:
        with self._conn() as conn:
            row = conn.execute(
                f"SELECT {_MEMORY_SELECT} FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
        return self._unpack(row) if row else None

    def get_all(self, memory_type: Optional[str] = None) -> List[MemoryEntry]:
        with self._conn() as conn:
            if memory_type:
                rows = conn.execute(
                    f"SELECT {_MEMORY_SELECT} FROM memories WHERE memory_type = ? ORDER BY timestamp DESC",
                    (memory_type,),
                ).fetchall()
            else:
                rows = conn.execute(f"SELECT {_MEMORY_SELECT} FROM memories ORDER BY timestamp DESC").fetchall()
        return [self._unpack(r) for r in rows]

    @staticmethod
    def _decode_embeddings(rows: Sequence[tuple]) -> Tuple[List[str], List[np.ndarray]]:
        ids: List[str] = []
        vectors: List[np.ndarray] = []
        for memory_id, blob in rows:
            try:
                vectors.append(np.asarray(pickle.loads(blob), dtype=np.float32).ravel())
            except Exception:
                continue  # one unreadable blob must not sink the whole recall
            ids.append(memory_id)
        return ids, vectors

    def find_by_similarity(
        self,
        embedding: np.ndarray,
        threshold: float,
        top_k: int,
        memory_type: Optional[str] = None,
    ) -> List[MemoryEntry]:
        if top_k <= 0:
            return []
        try:
            with self._conn() as conn:
                # Only embedded rows are scoreable, and only id+blob is needed to
                # rank them — the full rows are fetched for the winners alone.
                if memory_type:
                    rows = conn.execute(
                        "SELECT id, embedding FROM memories "
                        "WHERE embedding IS NOT NULL AND memory_type = ? ORDER BY timestamp DESC",
                        (memory_type,),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT id, embedding FROM memories WHERE embedding IS NOT NULL ORDER BY timestamp DESC"
                    ).fetchall()
                if not rows:
                    return []

                ids, vectors = self._decode_embeddings(rows)
                hits = [ids[i] for i in _rank_by_cosine(embedding, vectors, threshold, top_k)]
                if not hits:
                    return []

                placeholders = ",".join("?" * len(hits))
                full_rows = conn.execute(
                    f"SELECT {_MEMORY_SELECT} FROM memories WHERE id IN ({placeholders})",
                    hits,
                ).fetchall()

            by_id = {r[0]: self._unpack(r) for r in full_rows}
            return [by_id[memory_id] for memory_id in hits if memory_id in by_id]
        except Exception as e:
            logger.error(f"[SQLiteMemoryStore] similarity search failed: {e}")
            return []

    def remove(self, memory_id: str) -> bool:
        try:
            with self._conn() as conn:
                conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            return True
        except Exception as e:
            logger.error(f"[SQLiteMemoryStore] remove failed: {e}")
            return False

    def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        _allowed = {
            "content",
            "importance",
            "access_count",
            "last_accessed",
            "tags",
            "context",
            "embedding",
        }
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
            return True
        except Exception as e:
            logger.error(f"[SQLiteMemoryStore] update failed: {e}")
            return False

    def bump_access(self, memory_id: str, last_accessed: Optional[str] = None) -> bool:
        """Increment access_count in the database rather than via read-modify-write.

        Two recalls of the same memory from different threads each read the old
        count and write back old+1, so one of the two accesses disappears.
        """
        try:
            with self._conn() as conn:
                cur = conn.execute(
                    "UPDATE memories SET access_count = COALESCE(access_count, 0) + 1, "
                    "last_accessed = COALESCE(?, last_accessed) WHERE id = ?",
                    (last_accessed, memory_id),
                )
            return cur.rowcount > 0
        except Exception as e:
            logger.error(f"[SQLiteMemoryStore] bump_access failed: {e}")
            return False

    def count(self, memory_type: Optional[str] = None) -> int:
        with self._conn() as conn:
            if memory_type:
                return conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE memory_type = ?",
                    (memory_type,),
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
