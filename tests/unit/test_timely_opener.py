"""She opens with the one timely thing, when there is one, in her own words.

The opener replayed learned greeting phrases before ever asking the model, and
every model greeting was saved for replay, so openers turned into a rotation of
canned lines that knew nothing about his day. North star: surface something
only when it is timely and unambiguously relevant, which on most days means
nothing.
"""

from datetime import datetime
from types import SimpleNamespace

from ai.planning.proactive_assistant import pick_timely_note
from app.main import ALICE

NOW = datetime(2026, 9, 24, 9, 0)


def note(title, due_date, **fields):
    return SimpleNamespace(title=title, due_date=due_date, archived=False, checklist_items=None, **fields)


def test_the_freshest_overdue_note_is_raised():
    notes = [note("Old thing", "2026-09-19T10:00"), note("Call the bank", "2026-09-23T17:00")]
    assert pick_timely_note(notes, NOW) == 'the note "Call the bank" was due yesterday'


def test_a_note_overdue_for_weeks_is_abandoned_not_pressing():
    assert pick_timely_note([note("Old thing", "2026-08-01T10:00")], NOW) is None


def test_a_note_due_today_is_raised_with_its_time_when_it_has_one():
    assert pick_timely_note([note("Submit expenses", "2026-09-24T15:30")], NOW) == (
        'the note "Submit expenses" is due today at 15:30'
    )
    assert pick_timely_note([note("Submit expenses", "2026-09-24")], NOW) == 'the note "Submit expenses" is due today'


def test_finished_notes_are_never_raised():
    archived = note("Call the bank", "2026-09-23T17:00")
    archived.archived = True
    ticked = note("Groceries", "2026-09-23T17:00")
    ticked.checklist_items = [{"text": "milk", "checked": True}]
    assert pick_timely_note([archived, ticked], NOW) is None


def test_nothing_timely_means_nothing_is_raised():
    assert pick_timely_note([note("Someday", None), note("Next month", "2026-10-30")], NOW) is None


def _alice(timely):
    alice = ALICE.__new__(ALICE)
    alice._resolve_runtime_user_name = lambda: "Gabriel"
    alice._timely_note = lambda: timely
    alice.conversational_engine = SimpleNamespace(
        learned_greetings=["Hey.", "Yo."],
        _unique_candidates=lambda options: options,
        _pick_non_repeating=lambda options: options[0],
    )
    alice._learned_greeting_response = lambda **_kwargs: "Hey Gabriel."
    alice.prompts = []

    def request(**kwargs):
        alice.prompts.append(kwargs["prompt"])
        return SimpleNamespace(success=True, response="Morning. The bank call slipped to yesterday, still on it?")

    alice.llm_gateway = SimpleNamespace(request=request)
    alice.learned = []
    alice.phrasing_learner = SimpleNamespace(record_phrasing=lambda **kwargs: alice.learned.append(kwargs))
    return alice


def test_a_timely_note_is_raised_in_her_own_words_and_never_replayed():
    alice = _alice('the note "Call the bank" was due yesterday')
    greeting = alice._get_greeting()
    assert "bank" in greeting.lower()
    assert '"Call the bank"' in alice.prompts[0]
    assert alice.learned == []


def test_with_nothing_timely_the_opener_is_unchanged():
    alice = _alice(None)
    assert alice._get_greeting() == "Hey."
    assert alice.prompts == []
