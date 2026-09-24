"""Only feedback on how she talks becomes a standing preference.

Learned preferences go into every prompt after they are stored. The patterns
also matched one-off requests and passing words: "give me a brief overview"
was kept as "always be brief", "that's an elaborate plan" as "always go into
detail", and "write it in a professional tone" made every later reply formal.
"""

import pytest

from ai.runtime.companion_runtime import CompanionRuntimeLoop

learn = CompanionRuntimeLoop._preferences_in


@pytest.mark.parametrize(
    "text, expected",
    [
        ("that's way too long", {"response_length": "brief"}),
        ("you talk too much", {"response_length": "brief"}),
        ("keep your answers short", {"response_length": "brief"}),
        ("your answers are a bit too short", {"response_length": "detailed"}),
        ("you're too formal, relax", {"tone": "casual"}),
        ("no more bullets please", {"format": "prose"}),
        ("I prefer bullet points", {"format": "bullets"}),
        ("stop using emojis", {"emojis": "never"}),
    ],
)
def test_feedback_on_how_she_talks_is_kept(text, expected):
    assert learn(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "give me a brief overview of rust",
        "that's an elaborate plan",
        "can you explain more about the tier file?",
        "write the email in a professional tone",
        "is there a shorter route to work?",
        "go deeper on the second point",
        "I have no list for that",
    ],
)
def test_a_request_for_one_answer_is_not_a_standing_preference(text):
    assert learn(text) == {}
