import pytest

from ai.core.nlp_processor import NLPProcessor, ParsedCommand, RichToken


def _token(word: str) -> RichToken:
    return RichToken(
        text=word,
        normalized=word,
        kind="word",
        role="unknown",
        start_pos=0,
        end_pos=len(word),
        flags={},
    )


def test_plugin_score_counts_repeated_domain_terms():
    nlp = NLPProcessor()
    tokens = [_token("note"), _token("note"), _token("list")]

    scores = nlp._compute_plugin_scores(tokens, ParsedCommand())

    assert scores["notes"] == pytest.approx(3.6)


def test_weather_score_remains_dominant_with_forecast_signals():
    nlp = NLPProcessor()
    tokens = [
        _token("weather"),
        _token("forecast"),
        _token("rain"),
        _token("will"),
        _token("rain"),
    ]
    parsed = ParsedCommand(action="forecast", object_type="weather")

    scores = nlp._compute_plugin_scores(tokens, parsed)

    assert scores["weather"] > 9.0
    assert scores["weather"] > scores["notes"]


def test_wake_word_prefix_is_removed_before_tokenization():
    nlp = NLPProcessor()

    debug = nlp.debug_tokenizer("Hey assistant, what is the weather tomorrow?")

    assert not debug["normalized_text"].lower().startswith("assistant")
    assert debug["parsed_command"]["object_type"] == "weather"


def test_foundation_layer_metadata_is_attached_to_modifiers():
    nlp = NLPProcessor()

    result = nlp.process("create a note called sprint retrospective")
    modifiers = result.parsed_command.get("modifiers", {})

    assert "plan_memory" in modifiers
    assert "evaluation_harness" in modifiers
    assert "latency_budget" in modifiers
    assert "clarification_policy" in modifiers


def test_authorization_policy_blocks_high_risk_delete_language():
    nlp = NLPProcessor()

    result = nlp.process("delete all notes right now")
    modifiers = result.parsed_command.get("modifiers", {})
    auth = modifiers.get("authorization", {})

    assert result.intent == "conversation:clarification_needed"
    assert auth.get("requires_confirmation") is True
    assert modifiers.get("tool_execution_disabled") is True
