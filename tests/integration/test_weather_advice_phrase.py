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
