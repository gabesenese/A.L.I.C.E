from ai.core.nlp_processor import NLPProcessor


def test_weather_context_plus_reminder_request_prefers_reminder():
    nlp = NLPProcessor()
    result = nlp.process("nice weather today, do i have any reminders?")
    assert result.intent.startswith("reminder:")
    assert not result.intent.startswith("weather:")


def test_real_weather_question_stays_weather_current():
    nlp = NLPProcessor()
    result = nlp.process("what's the weather today?")
    assert result.intent == "weather:current"


def test_coat_question_stays_weather_current():
    nlp = NLPProcessor()
    result = nlp.process("should i wear a coat?")
    assert result.intent == "weather:current"


def test_which_day_coat_stays_weather_forecast():
    nlp = NLPProcessor()
    result = nlp.process("which day should i wear a coat?")
    assert result.intent == "weather:forecast"
