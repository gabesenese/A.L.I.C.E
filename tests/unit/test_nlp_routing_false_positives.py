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


@pytest.mark.parametrize(
    "text",
    [
        "remind me why we chose SQLite over Postgres",
        "remind me what you said about the scheduler",
        "remind me about the router design",
    ],
)
def test_remind_me_as_a_recall_question_is_a_question(nlp, text):
    assert nlp.process(text).intent == "conversation:question"


@pytest.mark.parametrize(
    "text",
    ["remind me to call mom at 5pm", "remind me about the dentist tomorrow", "set a reminder for 9am"],
)
def test_real_reminder_requests_still_are(nlp, text):
    assert nlp.process(text).intent == "reminder:set"
