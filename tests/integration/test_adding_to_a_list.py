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
