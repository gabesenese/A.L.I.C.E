"""Adding things to a list, the most ordinary thing to ask an assistant.

"add milk to my shopping list" went to notes:list because of the word "list",
and the append handler had no pattern for it either, so it failed: "I couldn't
get a result for that."
"""

import pytest

from ai.core.nlp_processor import NLPProcessor
from ai.plugins.notes_plugin import NotesManager, NotesPlugin


@pytest.fixture(scope="module")
def nlp():
    return NLPProcessor()


@pytest.mark.parametrize(
    "text",
    [
        "add milk to my shopping list",
        "add eggs and bread to the grocery list",
        "put batteries on my shopping list",
        "add call the dentist to my todo list",
        "can you add milk to my shopping list?",
    ],
)
def test_it_is_understood_as_adding(nlp, text):
    result = nlp.process(text)
    assert result.intent == "notes:append"
    # A question reads as conversation, and the gate that turns tools off for
    # conversation would send the turn to the model anyway.
    assert not result.parsed_command["modifiers"].get("tool_execution_disabled")


def test_showing_the_lists_is_still_showing(nlp):
    assert nlp.process("show me my lists").intent != "notes:append"


@pytest.fixture
def notes(tmp_path):
    plugin = NotesPlugin()
    plugin.manager = NotesManager(notes_dir=str(tmp_path))
    return plugin


def test_the_first_item_starts_the_list_and_the_next_ones_join_it(notes):
    first = notes._append_note("add milk to my shopping list")
    second = notes._append_note("add eggs, bread and butter to my shopping list")

    assert first["response"] == "Started a shopping list with milk."
    assert second["response"] == "Added eggs, bread and butter to your shopping list."
    [note] = notes.manager.find_by_title("shopping list")
    assert note.content.splitlines() == ["- milk", "- eggs", "- bread", "- butter"]


def test_both_ways_the_plugin_dispatches_it_add_the_item(notes):
    out = notes.execute("notes:append", "add milk to my shopping list", {}, {})

    assert out["success"] is True
    assert out["response"] == "Started a shopping list with milk."


def _say(notes, text, intent="notes:list"):
    return notes.execute(intent, text, {}, {})["response"]


def test_what_is_on_the_list_is_read_back(notes):
    """It counted every note instead: "You have 3 note(s)."."""
    _say(notes, "add milk, eggs and bread to my shopping list", "notes:append")
    notes.manager.create_note(title="Meeting notes", content="Talked about the roadmap.")

    assert _say(notes, "what's on my shopping list?") == "On your shopping list: milk, eggs and bread."
    assert _say(notes, "show me the shopping list") == "On your shopping list: milk, eggs and bread."


def test_things_come_off_the_list(notes):
    _say(notes, "add milk, eggs and bread to my shopping list", "notes:append")

    assert _say(notes, "take the milk off my shopping list") == "Took milk off your shopping list."
    assert _say(notes, "remove egg and cheese from the shopping list") == (
        "Took eggs off your shopping list; I couldn't find cheese on it."
    )
    assert _say(notes, "what's on my shopping list?") == "On your shopping list: bread."


def test_clearing_the_list_says_what_was_on_it(notes):
    _say(notes, "add milk and eggs to my shopping list", "notes:append")

    assert _say(notes, "clear my shopping list") == "Cleared your shopping list. It had milk and eggs."
    assert _say(notes, "what's on my shopping list?") == "Your shopping list is empty."


def test_a_list_that_does_not_exist_is_not_invented(notes):
    notes.manager.create_note(title="Meeting notes", content="Talked about the roadmap.")

    assert _say(notes, "what's on my packing list?") == "You don't have a packing list yet."


def test_an_unnamed_list_with_several_candidates_is_asked_about(notes):
    _say(notes, "add milk to my shopping list", "notes:append")
    _say(notes, "add call the dentist to my todo list", "notes:append")

    assert _say(notes, "what's on my list?") == "Which one: your todo list or your shopping list?"
