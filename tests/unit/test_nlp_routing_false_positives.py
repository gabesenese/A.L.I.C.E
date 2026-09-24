import pytest

from ai.core.nlp_processor import NLPProcessor


@pytest.fixture(scope="module")
def nlp():
    return NLPProcessor()


@pytest.mark.parametrize(
    "text",
    [
        "how much memory does a python dict use",
        "is the network layer in a transformer trainable",
        "what's in my memory",
        "this list is long",
    ],
)
def test_questions_that_mention_a_resource_word_are_not_system_status(nlp, text):
    assert nlp.process(text).intent != "system:status"


@pytest.mark.parametrize(
    "text",
    ["how is my cpu doing", "check disk space", "battery level", "what ports are listening"],
)
def test_questions_about_this_machine_still_are(nlp, text):
    assert nlp.process(text).intent == "system:status"
