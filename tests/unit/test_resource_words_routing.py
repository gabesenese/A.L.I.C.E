"""A resource word is a system-status request only when it is about this machine."""

import pytest

from ai.core.nlp_processor import NLPProcessor


@pytest.fixture(scope="module")
def nlp():
    return NLPProcessor()


@pytest.mark.parametrize(
    "text",
    [
        "is the disk scheduler in linux still using CFQ",
        "can you check if the battery on my bike light is dead",
        "how much memory does a python dict use per entry",
    ],
)
def test_unbound_resource_words_do_not_ask_for_a_system_readout(nlp, text):
    assert nlp.process(text).intent != "system:status"


@pytest.mark.parametrize("text", ["how's my cpu doing", "check system status", "how much disk space is left"])
def test_questions_about_this_machine_still_do(nlp, text):
    assert nlp.process(text).intent == "system:status"
