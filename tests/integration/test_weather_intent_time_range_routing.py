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


def test_weather_tomorrow_promotes_clarification_to_forecast():
    alice = ALICE.__new__(ALICE)
    alice._think = lambda *_args, **_kwargs: None

    intent, confidence = ALICE._normalize_weather_intent_for_time_range(
        alice,
        user_input="whats the weather for tomorrow?",
        intent="conversation:clarification_needed",
        intent_confidence=0.62,
    )

    assert intent == "weather:forecast"
    assert confidence >= 0.9


def test_weather_current_with_bare_time_range_phrase_promotes_to_forecast():
    alice = ALICE.__new__(ALICE)
    alice._think = lambda *_args, **_kwargs: None

    intent, confidence = ALICE._normalize_weather_intent_for_time_range(
        alice,
        user_input="what about this week?",
        intent="weather:current",
        intent_confidence=0.73,
    )

    assert intent == "weather:forecast"
    assert confidence >= 0.9


def test_non_weather_tomorrow_does_not_promote_clarification_intent():
    alice = ALICE.__new__(ALICE)
    alice._think = lambda *_args, **_kwargs: None

    intent, confidence = ALICE._normalize_weather_intent_for_time_range(
        alice,
        user_input="can you remind me tomorrow about groceries?",
        intent="conversation:clarification_needed",
        intent_confidence=0.62,
    )

    assert intent == "conversation:clarification_needed"
    assert confidence == 0.62


def test_weather_domain_override_promotes_notes_mislabel_to_forecast():
    alice = ALICE.__new__(ALICE)
    alice.last_intent = "weather:current"
    alice.conversation_topics = ["weather:forecast"]

    intent, confidence, meta = ALICE._apply_weather_domain_override(
        alice,
        user_input="is it gonna snow by any chance?",
        intent="notes:read",
        intent_confidence=0.88,
        entities={"resolved_reference": "Weather forecast for Kitchener"},
        followup_meta={"domain": "weather", "was_followup": True},
    )

    assert meta.get("applied") is True
    assert intent == "weather:forecast"
    assert confidence >= 0.9


def test_weather_domain_override_does_not_fire_for_explicit_note_request():
    alice = ALICE.__new__(ALICE)
    alice.last_intent = "notes:list"
    alice.conversation_topics = ["notes:list"]

    intent, confidence, meta = ALICE._apply_weather_domain_override(
        alice,
        user_input="read my note about snow conditions",
        intent="notes:read",
        intent_confidence=0.90,
        entities={},
        followup_meta={"domain": "notes", "was_followup": True},
    )

    assert meta.get("applied") is False
    assert intent == "notes:read"
    assert confidence == 0.90
