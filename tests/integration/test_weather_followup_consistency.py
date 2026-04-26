from datetime import datetime, timedelta

from ai.core.reasoning_engine import EntityKind, WorldEntity
from app.main import ALICE


class _ReasoningEngineStub:
    def __init__(self, entities):
        self._entities = entities

    def get_entity(self, entity_id):
        return self._entities.get(entity_id)


def _make_alice_stub(entities):
    alice = ALICE.__new__(ALICE)
    alice.reasoning_engine = _ReasoningEngineStub(entities)
    alice.conversation_summary = [{"intent": "weather:current"}]
    alice._think = lambda *args, **kwargs: None
    alice._alice_direct_phrase = lambda _rtype, payload: payload
    alice._generate_natural_response = (
        lambda payload, _tone, _context, _user_input: payload
    )
    return alice


def test_clothing_followup_prefers_newest_weather_snapshot_and_yes_no_flag():
    now = datetime.now()

    # Simulate stale current weather (older, warm) and fresh forecast (newer, cold).
    stale_current = WorldEntity(
        id="current_weather",
        kind=EntityKind.TOPIC,
        label="Current weather",
        created_at=now - timedelta(minutes=20),
        data={
            "temperature": 8,
            "condition": "clear sky",
            "location": "Kitchener",
            "message_code": "weather:current",
        },
    )
    fresh_forecast = WorldEntity(
        id="weather_forecast",
        kind=EntityKind.TOPIC,
        label="Forecast weather",
        created_at=now - timedelta(minutes=1),
        data={
            "forecast": [
                {
                    "date": now.strftime("%Y-%m-%d"),
                    "low": -5,
                    "high": 0,
                    "condition": "light snow",
                }
            ],
            "location": "Kitchener",
            "message_code": "weather:forecast",
        },
    )

    alice = _make_alice_stub(
        {
            "current_weather": stale_current,
            "weather_forecast": fresh_forecast,
        }
    )

    result = alice._handle_weather_followup(
        "should i bring a scarf too or no?", "weather:current"
    )

    assert isinstance(result, dict)
    assert result.get("type") == "weather_advice"
    # Derived from fresh forecast low/high average, not stale current weather.
    assert result.get("temperature") == -2
    assert result.get("clothing_item") == "scarf"
    assert result.get("force_yes_no") is True


def test_umbrella_typo_followup_maps_to_umbrella_item():
    now = datetime.now()

    rainy_current = WorldEntity(
        id="current_weather",
        kind=EntityKind.TOPIC,
        label="Current weather",
        created_at=now,
        data={
            "temperature": 6,
            "condition": "light rain",
            "location": "Kitchener",
            "message_code": "weather:current",
        },
    )

    alice = _make_alice_stub({"current_weather": rainy_current})

    result = alice._handle_weather_followup(
        "should i bring an umbrela?", "weather:forecast"
    )

    assert isinstance(result, dict)
    assert result.get("type") == "weather_advice"
    assert result.get("clothing_item") == "umbrella"
    assert result.get("force_yes_no") is True


def test_rain_tomorrow_question_returns_yes_no_outlook_phrase():
    now = datetime.now()
    tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")

    fresh_forecast = WorldEntity(
        id="weather_forecast",
        kind=EntityKind.TOPIC,
        label="Forecast weather",
        created_at=now,
        data={
            "forecast": [
                {
                    "date": tomorrow,
                    "low": 0,
                    "high": 8,
                    "condition": "heavy drizzle",
                }
            ],
            "location": "Kitchener",
            "message_code": "weather:forecast",
        },
    )

    alice = _make_alice_stub({"weather_forecast": fresh_forecast})

    result = alice._handle_weather_followup(
        "is it gonna rain tomorrow?", "weather:forecast"
    )

    assert isinstance(result, str)
    low = result.lower()
    assert low.startswith("yes")
    assert "tomorrow" in low
    assert "drizzle" in low or "rain" in low


def test_tomorrow_appointment_rain_concern_triggers_outlook_reply():
    now = datetime.now()
    tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")

    fresh_forecast = WorldEntity(
        id="weather_forecast",
        kind=EntityKind.TOPIC,
        label="Forecast weather",
        created_at=now,
        data={
            "forecast": [
                {
                    "date": tomorrow,
                    "low": 1,
                    "high": 12,
                    "condition": "overcast",
                }
            ],
            "location": "Kitchener",
            "message_code": "weather:forecast",
        },
    )

    alice = _make_alice_stub({"weather_forecast": fresh_forecast})

    result = alice._handle_weather_followup(
        "hows the weather for tomorrow i have an appointment at 5:30 pm tomorrow and need to make sure its not raining",
        "weather:forecast",
    )

    assert isinstance(result, str)
    low = result.lower()
    assert low.startswith("no")
    assert "tomorrow" in low
    assert "appointment" in low


def test_time_range_followup_ignores_stale_stored_forecast_snapshot():
    class _StaleLiveStateStub:
        def latest_weather_forecast(self, **_kwargs):
            return {
                "forecast": [
                    {
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "low": 1,
                        "high": 12,
                        "condition": "overcast",
                    }
                ],
                "location": "Kitchener",
                "is_stale": True,
            }

    alice = _make_alice_stub({})
    alice.live_state_service = _StaleLiveStateStub()

    result = alice._handle_weather_followup(
        "is it raining tomorrow?", "weather:forecast"
    )

    # Stale snapshot should be ignored so normal weather routing can fetch fresh data.
    assert result is None
