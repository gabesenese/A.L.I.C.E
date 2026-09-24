"""Correcting Alice invalidates what she had wrong, not what he just told her.

The correction ran after the turn's new facts were stored, and it marks the most
recent fact incorrect, so "that's wrong, my sister is called Anna" stored Anna,
then marked Anna incorrect, and left the old fact standing.
"""

import pytest

from ai.memory.memory_system import MemorySystem
from ai.memory.personal_memory import PersonalMemoryStore
from ai.runtime.alice_contract_factory import build_runtime_boundaries
from ai.runtime.memory_turn_service import MemoryTurnService
from tests.integration.test_contract_pipeline import _FakeAlice


@pytest.fixture
def setup(tmp_path):
    alice = _FakeAlice()
    alice.memory = MemorySystem(data_dir=str(tmp_path))
    boundaries = build_runtime_boundaries(alice)
    store = PersonalMemoryStore(alice.memory)
    store.store_structured_memory(
        content="my sister's name is Ana",
        domain="personal_life",
        kind="personal_fact",
        scope="long_term",
        confidence=0.95,
        source="explicit_request",
    )
    return alice, boundaries, store


def _correct(boundaries, text):
    service = MemoryTurnService()
    plan = service.build_memory_plan(
        user_input=text,
        user_name="Gabriel",
        trace_id="t1",
        decision_intent="conversation:general",
        decision_route="llm",
        episodic_payload={},
    )
    service.store_memory_plan(boundaries=boundaries, plan=plan)


def _valid(store):
    return [
        r["content"]
        for r in store.find_recent_structured_memories(top_k=20)
        if not (r.get("context") or {}).get("invalid")
    ]


def test_the_old_fact_is_the_one_marked_wrong(setup):
    _alice, boundaries, store = setup

    _correct(boundaries, "that's wrong, my sister is called Anna")

    assert "my sister's name is Ana" not in _valid(store)


def test_a_second_correction_does_not_land_on_the_same_fact_again(setup):
    _alice, boundaries, store = setup
    store.store_structured_memory(
        content="my dentist is Dr. Silva",
        domain="personal_life",
        kind="personal_fact",
        scope="long_term",
        confidence=0.95,
        source="explicit_request",
    )

    _correct(boundaries, "that's wrong")
    _correct(boundaries, "that's wrong")

    assert _valid(store) == []


def test_a_corrected_fact_stops_coming_back(setup):
    """Marking a fact invalid did not keep it out of recall: search results carry
    no context, so the flag was invisible and the wrong fact kept reaching the
    prompt, and the answer."""
    from ai.contracts import MemoryRequest
    from ai.plugins.memory_plugin import MemoryPlugin

    alice, boundaries, store = setup
    _correct(boundaries, "that's wrong")

    recalled = boundaries.memory.recall(MemoryRequest(query="sister", user_id="u1", max_items=5))
    assert "my sister's name is Ana" not in [i.get("content") for i in recalled.items]

    answer = MemoryPlugin(memory_system=alice.memory).execute(
        "memory:recall", "what's my sister's name?", {"topic": "sister"}, {}
    )
    assert "Ana" not in answer["response"]


def test_a_correction_lands_on_the_fact_her_last_reply_used(setup):
    alice, boundaries, store = setup
    store.store_structured_memory(
        content="my birthday is march 3rd",
        domain="personal_life",
        kind="personal_fact",
        scope="long_term",
        confidence=0.95,
        source="explicit_request",
    )
    alice.last_interaction = {"user_input": "what's my sister's name?", "assistant_response": "Your sister is Ana."}

    _correct(boundaries, "that's wrong")

    assert _valid(store) == ["my birthday is march 3rd"]


def test_saying_an_unrelated_answer_was_wrong_leaves_her_memory_alone(setup):
    """Told a weather answer was wrong, she used to invalidate whatever personal
    fact happened to be the newest."""
    alice, boundaries, store = setup
    alice.last_interaction = {"user_input": "weather?", "assistant_response": "12 degrees and raining in Kitchener."}

    _correct(boundaries, "that's wrong")

    assert _valid(store) == ["my sister's name is Ana"]
