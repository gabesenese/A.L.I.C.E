"""Padding gets stripped. A question does not.

Two layers were removing the trailing question from a reply. One was a filler
list that treated "What do you think?" as throat-clearing; the other stripped any
trailing question on conversational turns "so the take can stand" — but its guard
required more than one sentence, so it only ever fired when a take had already
been given and Alice then asked something back.

That is not a question diluting an answer. It is the reciprocity that makes an
exchange a conversation rather than a lookup returning a value, and removing it
is a large part of why a reply lands like output.
"""

import re

import pytest

from ai.runtime.response_discipline import apply_response_discipline, strip_filler_closing


# -- what still gets stripped -------------------------------------------------


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("I'd fix the schema drift first. Let me know if you need anything else.", "I'd fix the schema drift first."),
        ("I'd fix the schema drift first. Hope this helps!", "I'd fix the schema drift first."),
        ("I'd fix the schema drift first. Feel free to ask if anything is unclear.", "I'd fix the schema drift first."),
        ("I'd fix the schema drift first. I'm here to help.", "I'd fix the schema drift first."),
    ],
)
def test_servile_sign_offs_are_still_removed(reply, expected):
    assert strip_filler_closing(reply) == expected


def test_sycophantic_openings_are_still_removed():
    polished = apply_response_discipline("That's a great question! SQLite is fine here.")
    assert polished == "SQLite is fine here."


# -- what must survive --------------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        "I'd fix the schema drift first. What do you think?",
        "Rewriting costs you a week. What are your thoughts on doing it incrementally?",
        "I'd start with the memory layer. Does that match what you were thinking?",
        "SQLite is fine for one user. Are you expecting concurrent writers?",
    ],
)
def test_a_real_question_back_is_kept(reply):
    """An assistant that never asks anything is a lookup with a personality
    setting, not someone in a conversation."""
    assert apply_response_discipline(reply).rstrip().endswith("?"), reply


def test_a_take_followed_by_a_question_keeps_both():
    reply = "I'd fix the schema drift first. What do you think?"
    kept = apply_response_discipline(reply)
    assert "schema drift" in kept
    assert "What do you think?" in kept


# -- the inverted guard -------------------------------------------------------


def test_the_conversational_path_no_longer_strips_a_trailing_question():
    """The strip lived in the respond boundary and fired only when more than one
    sentence was present — that is, only after a take had been given."""
    import inspect

    from ai.runtime.boundaries import boundary_factory

    source = inspect.getsource(boundary_factory.build_runtime_boundaries)
    assert 'if _is_discussion and llm_text.endswith("?")' not in source


def test_a_question_only_reply_is_still_treated_as_a_deflection():
    """The case the old comment described — a question standing in *for* an
    answer — is caught by the retry gate, which regenerates the turn rather than
    deleting the question and leaving nothing."""
    import inspect

    from ai.runtime.boundaries import boundary_factory

    source = inspect.getsource(boundary_factory.build_runtime_boundaries)
    assert "_is_question_only" in source
    assert re.search(r"_is_hedge or _is_question_only", source)
