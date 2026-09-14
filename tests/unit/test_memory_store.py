"""Tests for the SQLite memory store: schema stability, connection hygiene, recall.

The regression that matters most here is schema drift. `hierarchical_compressor`
extends the `memories` table; when reads used `SELECT *` plus a fixed positional
unpack, the first compression pass turned every subsequent read into a
ValueError, which the loader swallowed — Alice booted with no memory at all and
said nothing about it.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import numpy as np
import pytest

from ai.memory.hierarchical_compressor import HierarchicalCompressor
from ai.memory.memory_store import (
    MEMORY_COLUMNS,
    MemoryEntry,
    SQLiteMemoryStore,
    ensure_memory_schema,
)


@pytest.fixture()
def db_path(tmp_path) -> str:
    return str(tmp_path / "memories.db")


@pytest.fixture()
def store(db_path) -> SQLiteMemoryStore:
    return SQLiteMemoryStore(db_path=db_path)


def _entry(mid: str, content: str = "content", **kwargs) -> MemoryEntry:
    fields = {
        "id": mid,
        "content": content,
        "memory_type": "episodic",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "context": {},
    }
    fields.update(kwargs)
    return MemoryEntry(**fields)


def _unit(*values: float) -> list:
    vec = np.array(values, dtype=np.float32)
    return (vec / np.linalg.norm(vec)).tolist()


class _ConnectionTracker:
    """Wrap sqlite3.connect so a test can see every connection and statement.

    Scoped to one database file: other tests leave background threads running
    that open databases of their own, and those are none of this test's business.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = str(db_path)
        self.opened: list = []
        self.statements: list = []

    def install(self, monkeypatch) -> None:
        real_connect = sqlite3.connect

        def _tracking_connect(database, *args, **kwargs):
            conn = real_connect(database, *args, **kwargs)
            if str(database) == self.db_path:
                conn.set_trace_callback(self.statements.append)
                self.opened.append(conn)
            return conn

        monkeypatch.setattr(sqlite3, "connect", _tracking_connect)

    @property
    def still_open(self) -> list:
        alive = []
        for conn in self.opened:
            try:
                conn.execute("SELECT 1")
            except sqlite3.ProgrammingError:
                continue  # closed, which is what we want
            alive.append(conn)
        return alive


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_add_then_get_by_id_returns_every_field(self, store):
        entry = _entry(
            "m1",
            "Gabriel runs 5km daily",
            importance=0.83,
            access_count=4,
            last_accessed="2026-02-02T09:00:00+00:00",
            embedding=_unit(1.0, 0.0, 0.0),
            tags=["fitness", "routine"],
            context={"source": "chat", "turn": 12},
            source_file="notes.md",
            chunk_index=3,
        )
        assert store.add(entry)

        loaded = store.get_by_id("m1")
        assert loaded is not None
        assert loaded.content == "Gabriel runs 5km daily"
        assert loaded.importance == pytest.approx(0.83)
        assert loaded.access_count == 4
        assert loaded.last_accessed == "2026-02-02T09:00:00+00:00"
        assert loaded.tags == ["fitness", "routine"]
        assert loaded.context == {"source": "chat", "turn": 12}
        assert loaded.source_file == "notes.md"
        assert loaded.chunk_index == 3
        assert loaded.embedding == pytest.approx([1.0, 0.0, 0.0])

    def test_get_by_id_missing_returns_none(self, store):
        assert store.get_by_id("nope") is None

    def test_get_all_filters_by_type_and_orders_newest_first(self, store):
        store.bulk_add(
            [
                _entry("old", timestamp="2026-01-01T00:00:00+00:00"),
                _entry("new", timestamp="2026-03-01T00:00:00+00:00"),
                _entry("fact", memory_type="semantic"),
            ]
        )
        assert [m.id for m in store.get_all("episodic")] == ["new", "old"]
        assert [m.id for m in store.get_all("semantic")] == ["fact"]
        assert store.count() == 3
        assert store.count("episodic") == 2

    def test_update_and_remove(self, store):
        store.add(_entry("m1"))
        assert store.update("m1", {"importance": 0.91, "tags": ["x"]})
        reloaded = store.get_by_id("m1")
        assert reloaded.importance == pytest.approx(0.91)
        assert reloaded.tags == ["x"]

        assert store.remove("m1")
        assert store.get_by_id("m1") is None

    def test_update_rejects_unknown_fields(self, store):
        store.add(_entry("m1"))
        assert store.update("m1", {"id": "hijacked"}) is False


# ---------------------------------------------------------------------------
# Schema drift
# ---------------------------------------------------------------------------


class TestSchemaDrift:
    def test_reads_survive_the_compressors_alter_table(self, store, db_path):
        store.add(_entry("m1", "before compression", embedding=_unit(1.0, 0.0, 0.0)))

        HierarchicalCompressor(db_path=Path(db_path))

        assert store.get_by_id("m1").content == "before compression"
        assert [m.id for m in store.get_all()] == ["m1"]
        assert [m.id for m in store.find_by_similarity(np.array(_unit(1.0, 0.0, 0.0)), 0.5, 5)] == ["m1"]

    def test_reads_survive_an_unrelated_added_column(self, store, db_path):
        store.add(_entry("m1"))
        with sqlite3.connect(db_path) as conn:
            conn.execute("ALTER TABLE memories ADD COLUMN some_future_field TEXT")

        assert store.get_by_id("m1") is not None
        assert store.count() == 1

    def test_compressor_columns_exist_on_a_fresh_store(self, store, db_path):
        with sqlite3.connect(db_path) as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(memories)")}
        assert {"memory_level", "parent_id"} <= columns

    def test_ensure_memory_schema_upgrades_a_pre_existing_table(self, tmp_path):
        """A database written before the compressor existed gains its columns, keeping its rows."""
        path = tmp_path / "legacy.db"
        with sqlite3.connect(path) as conn:
            conn.execute(
                "CREATE TABLE memories ("
                "id TEXT PRIMARY KEY, content TEXT NOT NULL, memory_type TEXT NOT NULL, "
                "timestamp TEXT NOT NULL, context TEXT DEFAULT '{}', importance REAL DEFAULT 0.5, "
                "access_count INTEGER DEFAULT 0, last_accessed TEXT, embedding BLOB, "
                "tags TEXT DEFAULT '[]', source_file TEXT, chunk_index INTEGER)"
            )
            conn.execute(
                "INSERT INTO memories (id, content, memory_type, timestamp) "
                "VALUES ('m1', 'legacy row', 'episodic', '2026-01-01T00:00:00+00:00')"
            )

        with sqlite3.connect(path) as conn:
            ensure_memory_schema(conn)
            columns = {row[1] for row in conn.execute("PRAGMA table_info(memories)")}
            assert {"memory_level", "parent_id"} <= columns
            assert conn.execute("SELECT content FROM memories WHERE id='m1'").fetchone()[0] == "legacy row"

    def test_unpack_ignores_trailing_columns(self):
        row = tuple(range(len(MEMORY_COLUMNS)))
        wide = ("m1", "text", "episodic", "2026-01-01", "{}", 0.5, 0, None, None, "[]", None, None, 1, "parent")
        assert len(wide) > len(row)
        assert SQLiteMemoryStore._unpack(wide).id == "m1"


# ---------------------------------------------------------------------------
# Connection hygiene
# ---------------------------------------------------------------------------


class TestConnectionHygiene:
    def test_no_connection_survives_a_read_write_cycle(self, store, db_path, monkeypatch):
        tracker = _ConnectionTracker(db_path)
        tracker.install(monkeypatch)

        for i in range(10):
            store.add(_entry(f"m{i}", embedding=_unit(1.0, float(i), 0.0)))
            store.get_by_id(f"m{i}")
            store.get_all()
            store.count()
            store.update(f"m{i}", {"importance": 0.6})
            store.find_by_similarity(np.array(_unit(1.0, 0.0, 0.0)), 0.0, 3)

        assert tracker.opened, "sqlite3.connect was never called — the tracker is not wired up"
        assert tracker.still_open == []

    def test_connection_closes_even_when_the_statement_fails(self, store, db_path, monkeypatch):
        tracker = _ConnectionTracker(db_path)
        tracker.install(monkeypatch)

        with pytest.raises(sqlite3.OperationalError):
            with store._conn() as conn:
                conn.execute("SELECT * FROM table_that_does_not_exist")

        assert tracker.still_open == []

    def test_failed_write_is_rolled_back(self, store):
        store.add(_entry("m1", "original"))
        with pytest.raises(sqlite3.OperationalError):
            with store._conn() as conn:
                conn.execute("UPDATE memories SET content = 'clobbered' WHERE id = 'm1'")
                conn.execute("UPDATE nonexistent SET x = 1")

        assert store.get_by_id("m1").content == "original"


# ---------------------------------------------------------------------------
# Similarity search
# ---------------------------------------------------------------------------


class TestFindBySimilarity:
    @pytest.fixture()
    def populated(self, store):
        store.bulk_add(
            [
                _entry("exact", embedding=_unit(1.0, 0.0, 0.0)),
                _entry("close", embedding=_unit(1.0, 0.2, 0.0)),
                _entry("far", embedding=_unit(0.3, 1.0, 0.0)),
                _entry("opposite", embedding=_unit(-1.0, 0.0, 0.0)),
                _entry("no_embedding"),
                _entry("semantic_hit", memory_type="semantic", embedding=_unit(1.0, 0.1, 0.0)),
            ]
        )
        return store

    def test_results_are_ranked_best_first(self, populated):
        hits = populated.find_by_similarity(np.array(_unit(1.0, 0.0, 0.0)), threshold=-1.0, top_k=10)
        ranked = [m.id for m in hits if m.memory_type == "episodic"]
        assert ranked == ["exact", "close", "far", "opposite"]

    def test_threshold_excludes_weak_matches(self, populated):
        hits = populated.find_by_similarity(np.array(_unit(1.0, 0.0, 0.0)), threshold=0.9, top_k=10)
        assert {m.id for m in hits} == {"exact", "close", "semantic_hit"}

    def test_top_k_caps_results(self, populated):
        hits = populated.find_by_similarity(np.array(_unit(1.0, 0.0, 0.0)), threshold=-1.0, top_k=2)
        assert [m.id for m in hits] == ["exact", "semantic_hit"]

    def test_memory_type_filter(self, populated):
        hits = populated.find_by_similarity(
            np.array(_unit(1.0, 0.0, 0.0)), threshold=0.0, top_k=10, memory_type="semantic"
        )
        assert [m.id for m in hits] == ["semantic_hit"]

    def test_rows_without_an_embedding_are_never_returned(self, populated):
        hits = populated.find_by_similarity(np.array(_unit(1.0, 0.0, 0.0)), threshold=-1.0, top_k=50)
        assert "no_embedding" not in {m.id for m in hits}

    def test_hits_carry_their_full_row(self, populated):
        hit = populated.find_by_similarity(np.array(_unit(1.0, 0.0, 0.0)), threshold=0.9, top_k=1)[0]
        assert hit.id == "exact"
        assert hit.content == "content"
        assert hit.embedding == pytest.approx([1.0, 0.0, 0.0])

    def test_empty_store_returns_empty(self, store):
        assert store.find_by_similarity(np.array(_unit(1.0, 0.0, 0.0)), threshold=0.0, top_k=5) == []

    def test_mismatched_dimensions_do_not_sink_the_recall(self, store):
        store.bulk_add(
            [
                _entry("wide", embedding=_unit(1.0, 0.0, 0.0, 0.0)),
                _entry("match", embedding=_unit(1.0, 0.0, 0.0)),
            ]
        )
        hits = store.find_by_similarity(np.array(_unit(1.0, 0.0, 0.0)), threshold=0.5, top_k=5)
        assert [m.id for m in hits] == ["match"]

    def test_scoring_is_not_one_query_per_row(self, populated, db_path, monkeypatch):
        """A recall must not read the table row by row — that is the scan this replaced."""
        tracker = _ConnectionTracker(db_path)
        tracker.install(monkeypatch)

        populated.find_by_similarity(np.array(_unit(1.0, 0.0, 0.0)), threshold=-1.0, top_k=3)

        selects = [s for s in tracker.statements if s.lstrip().upper().startswith("SELECT")]
        assert len(selects) == 2, f"expected a candidate scan plus one fetch, got: {selects}"
        assert len(tracker.opened) == 1


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_bump_access_does_not_lose_increments(self, store):
        store.add(_entry("m1"))

        barrier = threading.Barrier(8)

        def _hit() -> None:
            barrier.wait()
            store.bump_access("m1", last_accessed="2026-05-05T00:00:00+00:00")

        threads = [threading.Thread(target=_hit) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        reloaded = store.get_by_id("m1")
        assert reloaded.access_count == 8
        assert reloaded.last_accessed == "2026-05-05T00:00:00+00:00"

    def test_bump_access_on_missing_row_reports_false(self, store):
        assert store.bump_access("ghost") is False

    def test_concurrent_compression_claims_each_entry_once(self, store, db_path):
        """Two passes racing must not each write their own summary of the same entries."""
        store.bulk_add(
            [_entry(f"m{i}", f"memory {i}", timestamp=f"2026-01-{(i % 28) + 1:02d}T10:00:00") for i in range(210)]
        )
        compressor = HierarchicalCompressor(db_path=Path(db_path))

        barrier = threading.Barrier(2)
        created = []

        def _pass() -> None:
            barrier.wait()
            created.append(compressor.compress_level(0))

        threads = [threading.Thread(target=_pass) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert sorted(created)[0] == 0, "the losing pass should have found nothing left to claim"

        with sqlite3.connect(db_path) as conn:
            summaries = conn.execute("SELECT COUNT(*) FROM memories WHERE memory_type='semantic'").fetchone()[0]
            parents = conn.execute(
                "SELECT COUNT(DISTINCT parent_id) FROM memories WHERE parent_id IS NOT NULL AND parent_id != ''"
            ).fetchone()[0]
            orphans = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE memory_type='episodic' AND (parent_id IS NULL OR parent_id='')"
            ).fetchone()[0]

        assert summaries == sum(created)
        assert parents == summaries, "a summary exists whose source entries were re-parented by the other pass"
        assert orphans == 0
