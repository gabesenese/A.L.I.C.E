"""The cost of showing a model what to sound like.

ai/core/persona.py teaches the voice with nine labelled exchanges, because an 8B
imitates far better than it follows. That is a strong format prime, and the same
model will occasionally copy the scaffolding along with the shape: a reply
prefixed "Alice:", a hallucinated "Gabriel:" turn written after it, or — in the
tool loop, whose exemplars show the act step inline — the bracketed
"(you call list_workspace_files - 11 entries)" line itself.

The prompt bans all three in words, which is the kind of negative an 8B mostly
honours. This is the belt to that braces. It is deliberately exact rather than
clever: a stripper that guesses will eat real text, and the failure it prevents
is cosmetic while the failure it could cause is not.
"""

import pytest

from ai.runtime.response_discipline import apply_response_discipline, strip_speaker_label


# -- what gets removed --------------------------------------------------------


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("Alice: Eleven files, mostly the importer rewrite.", "Eleven files, mostly the importer rewrite."),
        ("alice: Good.", "Good."),
        ("  Alice:   Nothing I have.", "Nothing I have."),
        ("Assistant: SQLite is fine here.", "SQLite is fine here."),
    ],
)
def test_a_copied_speaker_label_is_removed(reply, expected):
    assert strip_speaker_label(reply) == expected


def test_a_bracketed_act_line_is_removed():
    """The tool exemplars show the act step so that what gets imitated is
    look-then-speak. The visible side effect is the model narrating it."""
    reply = "(you call list_workspace_files - 11 entries)\nAlice: Eleven files."
    assert strip_speaker_label(reply) == "Eleven files."


def test_a_hallucinated_next_turn_is_dropped():
    """Everything from a "Gabriel:" label onward is the model writing his side of
    the conversation, which is never part of the reply."""
    reply = "Eleven files.\n\nGabriel: thanks\nAlice: Good."
    assert strip_speaker_label(reply) == "Eleven files."


def test_only_the_first_label_is_treated_as_a_prefix():
    """Two labels means the model wrote a transcript; the first reply is hers."""
    assert strip_speaker_label("Alice: Good.\nGabriel: really?\nAlice: Really.") == "Good."


# -- what must survive --------------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        "The call: it depends on whether you have concurrent writers.",
        "Three bands by blast radius: reads, writes, and anything irreversible.",
        "Verdict: I'd leave it.",
        "He said something to me: that the schema had drifted.",
        "(I already checked, and it is empty.)",
        "Parenthetical (you can ignore this) in the middle of a sentence.",
    ],
)
def test_ordinary_text_is_left_alone(reply):
    assert strip_speaker_label(reply) == reply


def test_a_reply_that_is_only_a_label_does_not_become_empty():
    """An empty string reads to the caller as "the model had nothing to say",
    which triggers a fallback. Better to show the odd artefact than to erase a
    turn."""
    assert strip_speaker_label("Alice:") == "Alice:"


def test_an_empty_reply_stays_empty():
    assert strip_speaker_label("   ") == ""


# -- where it runs ------------------------------------------------------------


def test_the_discipline_pass_strips_labels_too():
    """Applied at the surface as well as at the source, so a reply assembled from
    somewhere other than chat() is still covered."""
    assert apply_response_discipline("Alice: SQLite is fine here.") == "SQLite is fine here."


def test_stripping_composes_with_the_filler_rules():
    assert apply_response_discipline("Alice: That's a great question! SQLite is fine here.") == "SQLite is fine here."


def test_the_engine_strips_before_it_records():
    """A leaked label left in the transcript re-primes the label on every later
    turn, so the recorded text has to be the cleaned text."""
    import inspect

    from ai.core.llm_engine import LocalLLMEngine

    source = inspect.getsource(LocalLLMEngine.chat)
    strip_at = source.index("strip_speaker_label")
    record_at = source.index("record_exchange")
    assert strip_at < record_at, "history records the raw reply, label and all"
