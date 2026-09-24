"""Asked to forget something, she forgets that, and only that.

"forget my favorite color" reached the model, which could only say it had
forgotten while the fact stayed in memory. The delete handler it should have
reached read its topic from entities the router never fills, and matched by
similarity, which counts "favorite food" as close to "favorite color".
"""

import pytest

from ai.memory.forgetting import as_his, forget_topic, is_about
from ai.memory.memory_extractor import MemoryExtractor
from ai.runtime.boundaries.boundary_factory import _as_told


@pytest.mark.parametrize(
    "text, topic",
    [
        ("forget my favorite color", "my favorite color"),
        ("please forget my address", "my address"),
        ("forget that my sister is Ana", "my sister is Ana"),
        ("forget what I told you about my job", "my job"),
        ("can you forget my birthday?", "my birthday"),
    ],
)
def test_the_request_names_what_to_forget(text, topic):
    assert forget_topic(text) == topic


@pytest.mark.parametrize(
    "text", ["forget it", "never mind, forget it", "forget about it", "forget that", "forget everything"]
)
def test_forget_it_means_never_mind(text):
    assert forget_topic(text) is None


def test_only_memories_about_it_match():
    assert is_about("User said: my favorite color is green", "my favorite color")
    assert not is_about("User said: my favorite food is sushi", "my favorite color")
    assert is_about("User said: my sister's name is Ana", "my sister is Ana")


def test_it_is_said_back_as_his():
    assert as_his("my favorite color") == "your favorite color"


def test_the_memory_plugin_forgets_what_was_named_and_keeps_the_rest(tmp_path):
    from ai.memory.memory_system import MemorySystem
    from ai.plugins.memory_plugin import MemoryPlugin

    memory = MemorySystem(data_dir=str(tmp_path))
    memory.store_memory(content="User said: my favorite color is green", memory_type="episodic")
    memory.store_memory(content="User said: my favorite food is sushi", memory_type="episodic")
    plugin = MemoryPlugin(memory)

    out = plugin.execute("memory:delete", "forget my favorite color", {}, {})
    left = [m["content"] for m in memory.get_all_memories(limit=50)]

    assert out["response"] == "Forgotten. I've deleted what you told me about your favorite color."
    assert left == ["User said: my favorite food is sushi"]
    again = plugin.execute("memory:delete", "forget my favorite color", {}, {})
    assert again["response"] == "I don't have anything saved about your favorite color."


@pytest.mark.parametrize(
    "text", ["what do you know about me?", "what do you know about me", "how are you", "is it raining"]
)
def test_a_question_is_not_stored_as_something_he_said_about_himself(text):
    assert not [c for c in MemoryExtractor().extract_from_user_turn(user_text=text) if c.should_store]


def test_a_fact_is_said_back_to_him_not_in_the_storage_format():
    assert _as_told("User said: my sister's name is Ana") == "Your sister's name is Ana"
    assert _as_told("Gabriel said: I'm allergic to peanuts\nAlice replied: Noted.") == "You're allergic to peanuts"
    assert _as_told("My boss said: no overtime") == "Your boss said: no overtime"


def test_the_heading_is_not_checked_as_a_claim():
    """Renamed to "Here's what I know about you:", the heading was read as an
    unsupported claim and every answer about him was refused."""
    from ai.memory.memory_answer_verifier import MemoryAnswerVerifier

    verdict = MemoryAnswerVerifier().verify_answer(
        answer_text="Here's what I know about you:\n- User said: my favorite food is sushi",
        evidence_items=[{"content": "User said: my favorite food is sushi"}],
    )

    assert verdict["accepted"] is True
