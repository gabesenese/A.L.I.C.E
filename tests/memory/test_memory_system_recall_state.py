"""Recall bookkeeping: the index must not double, and a failed load must be loud."""

import logging

import numpy as np
import pytest

from ai.memory.memory_system import MemorySystem, VectorStore


@pytest.fixture
def store():
    return VectorStore(dimension=3)


def _vec(*values):
    return np.array(values, dtype=np.float32)


# -- the vector index --------------------------------------------------------


def test_adding_the_same_id_twice_replaces_rather_than_duplicates(store):
    """The maintenance scheduler reloads memories on its own timer. Appending
    blindly meant every memory appeared twice after the first reload — and
    search is a linear scan, so recall got slower and returned duplicates."""
    store.add("m1", _vec(1, 0, 0), {"content": "first"})
    store.add("m1", _vec(0, 1, 0), {"content": "second"})

    assert store.ids == ["m1"]
    assert len(store.vectors) == 1
    assert store.metadata[0]["content"] == "second"


def test_distinct_ids_still_accumulate(store):
    store.add("m1", _vec(1, 0, 0), {"content": "a"})
    store.add("m2", _vec(0, 1, 0), {"content": "b"})
    assert store.ids == ["m1", "m2"]


def test_search_returns_each_memory_once_after_a_reload(store):
    for _ in range(3):
        store.add("m1", _vec(1, 0, 0), {"content": "only one"})
        store.add("m2", _vec(0, 1, 0), {"content": "the other"})

    hits = store.search(_vec(1, 0, 0), top_k=10)
    assert [hit[0] for hit in hits] == ["m1", "m2"]


def test_clear_empties_the_index(store):
    store.add("m1", _vec(1, 0, 0), {"content": "a"})
    store.clear()
    assert store.ids == []
    assert store.vectors == []
    assert store.metadata == []
    assert store.search(_vec(1, 0, 0)) == []


def test_a_dimension_mismatch_is_still_rejected(store):
    with pytest.raises(ValueError):
        store.add("m1", _vec(1, 0), {"content": "wrong width"})


# -- a load that fails -------------------------------------------------------


def test_a_failed_load_is_recorded_and_logged_as_an_error(monkeypatch, caplog):
    """An empty recall after a load failure is indistinguishable from a first
    run: Alice greets a long-time user as a stranger and nothing says why."""
    system = MemorySystem.__new__(MemorySystem)
    system.vector_store = VectorStore(dimension=3)
    system.load_failed = False

    def exploding_store():
        raise RuntimeError("no such table: memories")

    monkeypatch.setattr("ai.memory.memory_store.get_memory_store", exploding_store)

    with caplog.at_level(logging.ERROR, logger="ai.memory.memory_system"):
        system._load_memories()

    assert system.load_failed is True
    assert any(record.levelno >= logging.ERROR for record in caplog.records)


def test_the_index_is_cleared_before_a_reload(monkeypatch):
    system = MemorySystem.__new__(MemorySystem)
    system.vector_store = VectorStore(dimension=3)
    system.vector_store.add("stale", _vec(1, 0, 0), {"content": "from a previous load"})
    system.load_failed = False

    def exploding_store():
        raise RuntimeError("database is locked")

    monkeypatch.setattr("ai.memory.memory_store.get_memory_store", exploding_store)
    system._load_memories()

    # Even on the failure path the stale index must not survive and be mistaken
    # for current memory.
    assert system.vector_store.ids == []
