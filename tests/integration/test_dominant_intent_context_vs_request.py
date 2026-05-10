from ai.core.nlp_processor import NLPProcessor


def test_weather_context_plus_open_notes_request_prefers_notes():
    nlp = NLPProcessor()
    result = nlp.process(
        "going pretty good, weather is great today, looking forward to it, I was wondering if i have any open notes?"
    )
    assert result.intent in {"notes:query_exist", "notes:list"}
    assert not result.intent.startswith("weather:")


def test_cold_day_context_plus_work_on_alice_prefers_operator_continue():
    nlp = NLPProcessor()
    result = nlp.process(
        "good, it was a cold day today so i just stayed home, now i am gonna work on alice for a bit"
    )
    assert result.intent == "operator:continue"
    assert not result.intent.startswith("weather:")
