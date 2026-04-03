from app.main import ALICE


def _alice_stub():
    return ALICE.__new__(ALICE)


def test_weather_advice_umbrella_uses_an_and_clean_spelling():
    alice = _alice_stub()
    out = alice._alice_direct_phrase(
        "weather_advice",
        {
            "temperature": 6.1,
            "condition": "light drizzleing",
            "location": "Kitchener",
            "clothing_item": "umbrella",
            "force_yes_no": True,
        },
    )
    assert isinstance(out, str)
    assert out.startswith("Yes,")
    assert "an umbrella" in out
    assert "drizzling" in out


def test_weather_advice_coat_yes_no_grammar():
    alice = _alice_stub()
    out = alice._alice_direct_phrase(
        "weather_advice",
        {
            "temperature": -8,
            "condition": "clear",
            "clothing_item": "coat",
            "force_yes_no": True,
        },
    )
    assert out.startswith("Yes,")
    assert "a coat" in out


def test_weather_advice_unknown_condition_does_not_surface_unknown_text():
    alice = _alice_stub()
    out = alice._alice_direct_phrase(
        "weather_advice",
        {
            "temperature": None,
            "condition": "unknown",
            "location": "Kitchener",
            "clothing_item": "jacket",
            "force_yes_no": True,
        },
    )
    assert "unknown" not in out.lower()
    assert "conditions look cool" in out.lower()


def test_weather_forecast_weekend_unknown_condition_uses_friendly_label():
    alice = _alice_stub()
    out = alice._alice_direct_phrase(
        "weather_forecast",
        {
            "location": "Kitchener",
            "user_input": "what is the weather this weekend",
            "forecast": [
                {"date": "2030-06-01", "high": 22, "low": 13, "condition": "unknown"},
                {"date": "2030-06-02", "high": 21, "low": 12, "condition": "clear sky"},
            ],
        },
    )
    assert "Unknown" not in out
    assert "Conditions Unavailable" in out
