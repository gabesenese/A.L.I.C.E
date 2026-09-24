"""A newer fact about the same thing replaces the older one.

"Actually, my sister is called Anna" was stored next to "my sister's name is
Ana", and both stayed valid, so what she knew depended on which one recall
happened to rank first. Consolidation only merged near-identical wording.
"""

import pytest

from ai.memory.memory_system import MemorySystem
from ai.memory.personal_memory import PersonalMemoryStore


@pytest.fixture
def store(tmp_path):
    return PersonalMemoryStore(MemorySystem(data_dir=str(tmp_path)))


def _fact(store, content, kind="personal_fact"):
    return store.store_structured_memory(
        content=content,
        domain="personal_life",
        kind=kind,
        scope="long_term",
        confidence=0.9,
        source="conversation",
    )


def _known(store):
    stale = store.invalid_ids()
    return sorted({r["content"] for r in store._iter_structured_entries() if str(r.get("id")) not in stale})


def test_the_newer_fact_about_the_same_subject_wins(store):
    _fact(store, "my sister's name is Ana")
    _fact(store, "Gabriel said: my sister is called Anna", kind="relationship_context")

    assert _known(store) == ["Gabriel said: my sister is called Anna"]


def test_a_changed_date_replaces_the_old_one(store):
    _fact(store, "Gabriel said: my birthday is march 3rd")
    _fact(store, "Gabriel said: my birthday is march 4th")

    assert _known(store) == ["Gabriel said: my birthday is march 4th"]


def test_different_subjects_both_stand(store):
    _fact(store, "Gabriel said: my dentist appointment is on friday at 3pm")
    _fact(store, "Gabriel said: my dentist is Dr. Silva")
    _fact(store, "Gabriel said: my sister's birthday is june 5")
    _fact(store, "my sister's name is Ana")

    assert len(_known(store)) == 4
