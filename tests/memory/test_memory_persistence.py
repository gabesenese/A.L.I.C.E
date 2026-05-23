"""
Memory persistence tests — verify every mutation survives a simulated process restart.

Pattern per test:
  1. Wire singletons to a temp SQLite DB
  2. Perform an operation (store / forget / update / consolidate / remove / recall)
  3. Call restart() to wipe in-memory state and reload from the same SQLite file
  4. Assert the mutation is present in the reloaded state
"""

import os
import pytest

import ai.memory.memory_store as _store_mod
import ai.memory.memory_system as _system_mod
from ai.memory.memory_store import SQLiteMemoryStore
from ai.memory.memory_system import MemorySystem
from ai.memory.personal_memory import PersonalMemoryStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_path(tmp_path):
    return str(tmp_path / "test_alice.db")


@pytest.fixture(autouse=True)
def _clean_singletons():
    """Reset module-level singletons before and after every test."""
    _store_mod._memory_store = None
    _system_mod._memory_system = None
    yield
    _store_mod._memory_store = None
    _system_mod._memory_system = None


def _make_system(db_path: str, tmp_path) -> tuple:
    """Wire singletons to db_path and return (store, system)."""
    data_dir = str(tmp_path / "mem_data")
    os.makedirs(data_dir, exist_ok=True)
    store = SQLiteMemoryStore(db_path=db_path)
    _store_mod._memory_store = store  # must be set BEFORE MemorySystem()
    system = MemorySystem(data_dir=data_dir)
    _system_mod._memory_system = system
    return store, system


def _restart(db_path: str, tmp_path) -> tuple:
    """Simulate a process restart: wipe singletons, reload from same SQLite file."""
    _system_mod._memory_system = None
    _store_mod._memory_store = None
    return _make_system(db_path, tmp_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_write_through_survives_restart(db_path, tmp_path):
    """store_memory() writes to SQLite immediately; content is present after restart."""
    _, system = _make_system(db_path, tmp_path)
    mem_id = system.store_memory(
        "Gabriel runs 5km daily",
        memory_type="episodic",
        importance=0.8,
    )

    store2, system2 = _restart(db_path, tmp_path)

    row = store2.get_by_id(mem_id)
    assert row is not None, "Memory row missing from SQLite after restart"
    assert "5km" in row.content, f"Unexpected content: {row.content!r}"
    assert any(m.id == mem_id for m in system2.episodic_memory), (
        "Memory missing from reloaded in-memory list"
    )


def test_forget_marks_invalid_in_sqlite(db_path, tmp_path):
    """forget_recent_memory() sets context['invalid']=True in SQLite (does NOT delete the row)."""
    store, system = _make_system(db_path, tmp_path)
    pms = PersonalMemoryStore(system)
    mem_id = pms.store_structured_memory(
        "Gabriel prefers tea",
        domain="health",
        kind="preference",
        scope="personal",
        confidence=0.9,
        source="test",
    )

    pms.forget_recent_memory(domain="health", kind="preference")

    store2, _ = _restart(db_path, tmp_path)
    row = store2.get_by_id(mem_id)
    assert row is not None, "Row was deleted instead of marked invalid"
    assert (row.context or {}).get("invalid") is True, (
        f"context['invalid'] not set after restart; context={row.context}"
    )


def test_update_content_survives_restart(db_path, tmp_path):
    """update_memory() persists corrected content and context['corrected'] flag to SQLite."""
    store, system = _make_system(db_path, tmp_path)
    pms = PersonalMemoryStore(system)
    mem_id = pms.store_structured_memory(
        "Gabriel wakes at 7am",
        domain="routine",
        kind="habit",
        scope="personal",
        confidence=0.7,
        source="test",
    )

    pms.update_memory(mem_id, "Gabriel wakes at 6am", confidence=0.95)

    store2, _ = _restart(db_path, tmp_path)
    row = store2.get_by_id(mem_id)
    assert row is not None, "Updated memory row missing from SQLite after restart"
    assert "6am" in row.content, f"Old content still present: {row.content!r}"
    assert (row.context or {}).get("corrected") is True, (
        "context['corrected'] flag not persisted"
    )


def test_consolidation_superseded_survives_restart(db_path, tmp_path):
    """consolidate_recent() writes superseded=True to SQLite; flag survives restart."""
    store, system = _make_system(db_path, tmp_path)
    pms = PersonalMemoryStore(system)

    # Two near-identical strings — Jaccard similarity well above 0.68 threshold
    pms.store_structured_memory(
        "Gabriel runs 5km every morning",
        domain="fitness",
        kind="habit",
        scope="personal",
        confidence=0.7,
        source="test",
    )
    pms.store_structured_memory(
        "Gabriel runs 5km every morning routinely",
        domain="fitness",
        kind="habit",
        scope="personal",
        confidence=0.8,
        source="test",
    )

    store2, _ = _restart(db_path, tmp_path)
    all_rows = store2.get_all("episodic")
    superseded_count = sum(
        1 for r in all_rows if (r.context or {}).get("superseded") is True
    )
    assert superseded_count >= 1, (
        f"No superseded entries found after restart; total episodic rows={len(all_rows)}"
    )


def test_remove_deleted_from_sqlite(db_path, tmp_path):
    """_remove_memory_by_id() hard-deletes from SQLite; row absent after restart."""
    store, system = _make_system(db_path, tmp_path)
    mem_id = system.store_memory(
        "Temporary memory to delete",
        memory_type="episodic",
        importance=0.1,
    )
    assert store.get_by_id(mem_id) is not None, "Setup: memory not written to SQLite"

    system._remove_memory_by_id(mem_id)

    store2, system2 = _restart(db_path, tmp_path)
    assert store2.get_by_id(mem_id) is None, (
        "Deleted memory still present in SQLite after restart"
    )
    assert not any(m.id == mem_id for m in system2.episodic_memory), (
        "Deleted memory reloaded into in-memory list"
    )


def test_access_count_accumulates_in_sqlite(db_path, tmp_path):
    """recall_memory() increments access_count in SQLite; count survives restart."""
    store, system = _make_system(db_path, tmp_path)
    mem_id = system.store_memory(
        "Gabriel enjoys hiking on weekends",
        memory_type="episodic",
        importance=0.9,
    )

    # min_similarity=0.0 ensures the recall fires regardless of embedding quality
    system.recall_memory("hiking", top_k=5, min_similarity=0.0)
    system.recall_memory("hiking", top_k=5, min_similarity=0.0)

    store2, _ = _restart(db_path, tmp_path)
    row = store2.get_by_id(mem_id)
    assert row is not None, "Memory missing after restart"
    assert row.access_count >= 1, f"access_count not persisted; got {row.access_count}"
