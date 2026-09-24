"""/correct changes how Alice routes, from the next turn on.

Nothing recorded the last interaction on the live path, so /correct and
/feedback always said there was nothing to correct. And an intent correction
went into a store nothing applied: it changed nothing until an offline training
script ran, which wrote where nothing reads.
"""

import json

import pytest

from ai.core.nlp_processor import NLPProcessor
from app.main import ALICE
from tests.integration.test_every_turn_is_in_the_transcript import alice as alice  # noqa: F401  (fixture)
from ai.runtime.turn_orchestrator import run_default_turn

PHRASE = "is there anything from the dentist"


def test_every_turn_leaves_something_to_correct(alice):  # noqa: F811
    alice.replies.append("Nothing from the dentist.")

    shown = run_default_turn(alice, PHRASE)

    assert alice.last_interaction["user_input"] == PHRASE
    assert alice.last_interaction["assistant_response"] == shown
    assert alice.last_interaction["intent"]


@pytest.fixture
def corrected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    bare = ALICE.__new__(ALICE)
    bare.nlp = NLPProcessor()
    return bare


def test_a_corrected_intent_is_used_on_the_next_turn(corrected):
    assert corrected.nlp.process(PHRASE).intent != "notes:search"

    assert corrected._learn_intent_correction(PHRASE, "notes:search") is True

    assert corrected.nlp.process(PHRASE).intent == "notes:search"


def test_the_correction_is_kept_for_the_next_session(corrected, tmp_path):
    corrected._learn_intent_correction(PHRASE, "notes:search")
    corrected._learn_intent_correction(PHRASE, "notes:list")

    saved = json.loads((tmp_path / "memory" / "curated_patterns.json").read_text())
    assert saved["corrections"] == [
        {"user_input": PHRASE, "expected_intent": "notes:list", "source": "user_correction"}
    ]
    assert NLPProcessor().process(PHRASE).intent == "notes:list"


def test_free_text_is_recorded_but_does_not_reroute(corrected):
    assert corrected._learn_intent_correction(PHRASE, "it should have looked at my notes") is False
    assert corrected.nlp.process(PHRASE).intent != "notes:search"
