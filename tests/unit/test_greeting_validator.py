"""The greeting validator rejects fabrications, not ordinary ways of saying hello."""

import pytest

from ai.runtime.greeting_surface_policy import validate_greeting_candidate


def _check(candidate, time_period="morning"):
    return validate_greeting_candidate(
        candidate=candidate,
        user_input="hey",
        time_period=time_period,
        allow_focus_reference=False,
        user_name="Gabriel",
    )


@pytest.mark.parametrize(
    "candidate",
    ["Hey Gabriel. What's up?", "Hey. Something on your mind?", "Hey Gabriel", "Morning.", "Hey."],
)
def test_natural_greetings_are_accepted(candidate):
    result = _check(candidate)
    assert result.valid, result.reasons


@pytest.mark.parametrize(
    "candidate",
    [
        "Hello Gabriel, how's the Django project going?",  # a topic nothing recalled
        "Hey Gabriel. You seem stressed.",  # a feeling nobody reported
        "Welcome back, Gabriel.",  # a return nobody recorded
    ],
)
def test_fabricated_greetings_are_rejected(candidate):
    assert not _check(candidate).valid


def test_greeting_knows_the_time_of_day_without_being_told():
    from datetime import datetime

    from ai.runtime.greeting_surface_policy import render_grounded_greeting

    hour = datetime.now().astimezone().hour
    word = (
        "morning"
        if 5 <= hour <= 11
        else "afternoon"
        if 12 <= hour <= 16
        else "evening"
        if 17 <= hour <= 21
        else "night"
    )
    greeting = render_grounded_greeting(
        user_name="Gabriel",
        operator_state={},
        session_state={},
        user_input="hey",
        llm_generate=lambda prompt=None, **_: f"Good {word}, Gabriel.",
    )

    assert greeting.text == f"Good {word}, Gabriel."
