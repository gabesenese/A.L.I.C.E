"""One unreadable memory row must cost one memory, not all of them.

Taken from a real failure. A single episodic row in data/memory/alice.db had NUL
padding bleed into it from a damaged page: a zeroed embedding blob, a context
string carrying control characters, and tags of b"\\x00\\x00". Its content and
timestamp were intact and perfectly readable.

Unpacking that row raised, the raise escaped the list comprehension in get_all,
and MemorySystem._load_memories retried twice and gave up. Alice then ran the
whole session with recall disabled, and the log line said memories were "NOT
lost" while nothing could reach them. One bad row cost 466 good ones.
"""

import json
import pickle
import sqlite3

import numpy as np
import pytest

from ai.memory.memory_store import MemoryEntry, SQLiteMemoryStore


def _entry(memory_id: str, content: str) -> MemoryEntry:
    return MemoryEntry(
        id=memory_id,
        content=content,
        memory_type="episodic",
        timestamp="2026-08-09T16:59:27.096022",
        context={"content": content},
        importance=0.5,
        access_count=0,
        last_accessed=None,
        embedding=[0.25] * 8,
        tags=["real"],
    )


@pytest.fixture
def store(tmp_path):
    made = SQLiteMemoryStore(str(tmp_path / "alice.db"))
    for index in range(3):
        made.add(_entry(f"episodic_good_{index}", f"user=question {index}"))
    return made


def _damage(store: SQLiteMemoryStore, memory_id: str, **columns) -> None:
    """Write raw bytes straight past the encoder, the way a bad page would."""
    assignments = ", ".join(f"{name} = ?" for name in columns)
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            f"INSERT OR REPLACE INTO memories (id, content, memory_type, timestamp) "
            f"VALUES (?, 'user=what is the weather like today', 'episodic', '2026-08-09T16:59:27')",
            (memory_id,),
        )
        conn.execute(f"UPDATE memories SET {assignments} WHERE id = ?", (*columns.values(), memory_id))


def test_a_zeroed_embedding_costs_its_own_field_not_the_row(store):
    _damage(store, "episodic_damaged", embedding=b"\x00" * 8)

    loaded = {entry.id: entry for entry in store.get_all("episodic")}

    assert len(loaded) == 4, "the damaged row should still be readable"
    assert loaded["episodic_damaged"].embedding is None
    assert loaded["episodic_damaged"].content == "user=what is the weather like today"


def test_a_context_full_of_control_characters_does_not_lose_the_memory(store):
    _damage(store, "episodic_damaged", context='{"content": "x' + "\x00" * 200 + '"}')

    loaded = {entry.id: entry for entry in store.get_all("episodic")}

    assert loaded["episodic_damaged"].context == {}
    assert loaded["episodic_damaged"].content == "user=what is the weather like today"


def test_every_good_memory_survives_a_neighbour_that_cannot_be_read(store):
    _damage(
        store,
        "episodic_damaged",
        embedding=b"\x00" * 8,
        context='{"broken": "' + "\x00" * 50 + '"}',
        tags=b"\x00\x00",
    )

    loaded = store.get_all("episodic")

    assert len(loaded) == 4
    assert {entry.id for entry in loaded} >= {f"episodic_good_{i}" for i in range(3)}
    assert all(entry.content for entry in loaded)


def test_recall_still_works_alongside_a_damaged_row(store):
    """The failure that hurt: similarity ran inside a try that returned []."""
    _damage(store, "episodic_damaged", embedding=b"\x00" * 8)

    hits = store.find_by_similarity(np.array([0.25] * 8, dtype=np.float32), threshold=0.0, top_k=10)

    assert hits, "one unreadable embedding must not empty the whole similarity result"
    assert all(entry.id != "episodic_damaged" for entry in hits)


def test_an_undamaged_store_is_unchanged(store):
    loaded = store.get_all("episodic")

    assert len(loaded) == 3
    assert all(entry.embedding is not None for entry in loaded)
    assert all(entry.tags == ["real"] for entry in loaded)
    assert all(entry.context.get("content") for entry in loaded)


def test_a_row_damaged_beyond_field_level_is_skipped_not_fatal():
    """The backstop, for damage no single field can absorb.

    A NOT NULL constraint stops a row this broken being inserted, but a damaged
    page hands one back anyway, so _unpack_all is exercised directly here rather
    than through a store that would refuse to hold the row.
    """
    good = (
        "episodic_good",
        "user=question",
        "episodic",
        "2026-08-09T16:59:27",
        "{}",
        0.5,
        0,
        None,
        None,
        "[]",
        None,
        None,
    )
    truncated = ("episodic_gutted", "user=half a row")

    entries = SQLiteMemoryStore._unpack_all([good, truncated, good])

    assert [entry.id for entry in entries] == ["episodic_good", "episodic_good"]


def test_the_encoder_still_round_trips_what_it_wrote(store):
    original = _entry("episodic_round_trip", "user=does this survive")
    store.add(original)

    back = {entry.id: entry for entry in store.get_all("episodic")}["episodic_round_trip"]

    assert back.content == original.content
    assert back.tags == original.tags
    assert back.context == original.context
    assert pytest.approx(back.embedding, rel=1e-5) == original.embedding
    assert json.loads(json.dumps(back.context)) == original.context
    assert pickle.loads(pickle.dumps(np.array(back.embedding))).shape == (8,)
