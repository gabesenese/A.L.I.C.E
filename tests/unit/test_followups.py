"""Short follow-ups continue the turn before them.

"what about tomorrow?" after the agenda, "and eggs" after adding milk, and
"actually make it 6" after setting a reminder each went to the model on their
own, which had nothing to act with. With a note in context, "and what time is
it?" was even answered by the notes plugin, because it contains "it".
"""

from datetime import datetime

import pytest

from ai.core.followups import DAY_FOLLOWUP_RE, RESCHEDULE_RE, more_items
from ai.planning.reminders import ReminderStore
from ai.plugins.reminder_plugin import ReminderPlugin

NOW = datetime(2026, 9, 24, 14, 30)


@pytest.mark.parametrize(
    "text, items",
    [
        ("and eggs", "eggs"),
        ("also bread and butter", "bread and butter"),
        ("eggs too", "eggs"),
        ("plus coffee", "coffee"),
    ],
)
def test_more_items_are_picked_out(text, items):
    assert more_items(text) == items


@pytest.mark.parametrize("text", ["and what's the weather", "and then?", "thanks", "that too", "and it's late"])
def test_other_short_turns_are_not_items(text):
    assert more_items(text) is None


def test_day_and_reschedule_followups_are_recognised():
    assert DAY_FOLLOWUP_RE.match("what about tomorrow?")
    assert DAY_FOLLOWUP_RE.match("and this week")
    assert not DAY_FOLLOWUP_RE.match("tomorrow I have a test")
    assert RESCHEDULE_RE.match("actually make it 6").group("when") == "6"
    assert RESCHEDULE_RE.match("change it to tomorrow at 9").group("when") == "tomorrow at 9"


def test_make_it_6_moves_the_reminder_just_set(tmp_path):
    store = ReminderStore(tmp_path / "r.json")
    plugin = ReminderPlugin(store)
    plugin._last_set_id = store.add("call mom", NOW.replace(hour=17, minute=0)).id

    reply = plugin._reschedule("6", NOW)["response"]

    assert reply == "Moved it. I'll remind you to call mom at 6:00 PM."
    [reminder] = store.pending()
    assert reminder.due_at == NOW.replace(hour=18, minute=0)


def test_with_nothing_just_set_there_is_nothing_to_move(tmp_path):
    assert ReminderPlugin(ReminderStore(tmp_path / "r.json"))._reschedule("6", NOW) is None


def test_and_eggs_goes_on_the_list_just_added_to(tmp_path):
    from ai.plugins.notes_plugin import NotesManager, NotesPlugin

    notes = NotesPlugin()
    notes.manager = NotesManager(notes_dir=str(tmp_path))
    notes.execute("notes:append", "add milk to my shopping list", {}, {})

    assert notes.execute("notes:append", "and eggs", {}, {})["response"] == "Added eggs to your shopping list."


def test_follow_ups_keep_the_intent_of_the_turn_before():
    from ai.core.nlp_processor import NLPProcessor

    nlp = NLPProcessor()
    for first, then, expected in [
        ("what's on my schedule today?", "what about tomorrow?", "reminder:agenda"),
        ("add milk to my shopping list", "and eggs", "notes:append"),
        ("remind me at 5pm to call mom", "actually make it 6", "reminder:set"),
    ]:
        nlp.process(first)
        assert nlp.process(then).intent == expected, then


def test_make_it_6_is_not_sent_for_clarification():
    from ai.reference_resolver import ReferenceResolver

    assert ReferenceResolver._NOT_A_REFERENCE.search("actually make it 6")


def test_the_plugin_an_intent_names_wins_over_one_that_liked_the_words():
    from ai.plugins.plugin_system import PluginInterface, PluginManager

    class Greedy(PluginInterface):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def initialize(self):
            return True

        def can_handle(self, intent, entities, query=None):
            return True

        def execute(self, intent, query, entities, context):
            return {"success": True, "response": self.name}

        def shutdown(self):
            return None

    manager = PluginManager(plugins_dir="none", use_semantic=False)
    manager.register_plugin(Greedy("Notes Plugin"))
    manager.register_plugin(Greedy("TimePlugin"))

    assert manager.execute_for_intent("time:current", "and what time is it?", {}, {})["response"] == "TimePlugin"
