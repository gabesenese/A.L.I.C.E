"""Regression tests for weather time-range intent normalization."""

from app.main import ALICE


def test_weather_rest_of_week_promotes_current_to_forecast():
    alice = ALICE.__new__(ALICE)
    alice._think = lambda *_args, **_kwargs: None

    intent, confidence = ALICE._normalize_weather_intent_for_time_range(
        alice,
        user_input="hows the weather for the rest of the week?",
        intent="weather:current",
        intent_confidence=0.88,
    )

    assert intent == "weather:forecast"
    assert confidence >= 0.9


def test_weather_today_stays_current():
    alice = ALICE.__new__(ALICE)
    alice._think = lambda *_args, **_kwargs: None

    intent, confidence = ALICE._normalize_weather_intent_for_time_range(
        alice,
        user_input="hows the weather today?",
        intent="weather:current",
        intent_confidence=0.88,
    )

    assert intent == "weather:current"
    assert confidence == 0.88
