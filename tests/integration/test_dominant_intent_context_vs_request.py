from ai.core.nlp_processor import NLPProcessor
from ai.runtime.perception_frame import build_perception_frame


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


def test_actual_request_beats_casual_weather_keyword():
    nlp = NLPProcessor()
    frame = build_perception_frame(
        "weather is great today, I was wondering if i have any open notes?"
    )
    result = nlp.process(frame.actual_request or "weather is great today")
    assert result.intent in {"notes:query_exist", "notes:list"}
