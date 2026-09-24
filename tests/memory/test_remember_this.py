"""Asked to remember something, Alice stores the thing itself.

The store action looked for the fact in entities the router never fills and in
a context key it never sends, so every "remember that ..." came back "No content
to store", which the user saw as "I couldn't get a result for that."
"""

import pytest

from ai.memory.memory_system import MemorySystem
from ai.plugins.memory_plugin import MemoryPlugin


@pytest.fixture
def plugin(tmp_path):
    return MemoryPlugin(memory_system=MemorySystem(data_dir=str(tmp_path)))


@pytest.mark.parametrize(
    "request_text, fact",
    [
        ("remember that my sister's name is Ana", "my sister's name is Ana"),
        ("remember my sister is called Ana", "my sister is called Ana"),
        ("please remember that I prefer tea over coffee", "I prefer tea over coffee"),
        ("save this: my wifi password is on the fridge", "my wifi password is on the fridge"),
        ("don't forget that the meeting moved to 4pm", "the meeting moved to 4pm"),
    ],
)
def test_the_fact_is_stored_without_the_command(plugin, request_text, fact):
    result = plugin.execute("memory:store", request_text, {}, {})

    assert result["success"] is True
    assert [m["content"] for m in plugin.memory.get_all_memories(limit=10)] == [fact]


def test_it_can_be_recalled_by_name_afterwards(plugin):
    plugin.execute("memory:store", "remember that my sister's name is Ana", {}, {})

    result = plugin.execute("memory:recall", "what's my sister's name?", {"topic": "sister"}, {})

    assert "Ana" in result["response"]


def test_what_he_asked_to_be_remembered_is_what_she_knows_about_him(tmp_path, monkeypatch):
    """Stored as a plain memory, the fact was invisible to the personal-memory
    path: right after it was saved, "what do you know about me?" was answered "I do
    not have enough saved memory yet to answer that accurately"."""
    from ai.runtime.alice_contract_factory import build_runtime_boundaries
    from ai.runtime.contract_pipeline import ContractPipeline
    from tests.integration.test_contract_pipeline import _FakeAlice

    alice = _FakeAlice()
    alice.memory = MemorySystem(data_dir=str(tmp_path))
    MemoryPlugin(memory_system=alice.memory).execute("memory:store", "remember that my sister's name is Ana", {}, {})

    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn(
        user_input="what do you know about me?", user_id="u1", turn_number=2
    )

    assert "Ana" in result.response_text
