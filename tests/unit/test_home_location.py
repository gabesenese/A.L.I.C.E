"""She remembers where he lives, and stops asking.

Asked "What city should I check the weather for?", "Toronto" went to the model
as conversation, and the next weather question asked for the city again. "I
live in Toronto" was kept nowhere the weather could use it.
"""

import time

import pytest

from ai.identity.home_location import home_location, place_answer, remember_home, stated_home


@pytest.mark.parametrize(
    "text, place",
    [
        ("Toronto", "Toronto"),
        ("new york", "New York"),
        ("in Paris", "Paris"),
        ("Kitchener, Canada", "Kitchener, Canada"),
    ],
)
def test_a_place_name_answers_what_city(text, place):
    assert place_answer(text) == place


@pytest.mark.parametrize("text", ["thanks", "never mind", "it is raining", "why?", "is it cold"])
def test_other_replies_are_not_a_city(text):
    assert place_answer(text) is None


@pytest.mark.parametrize(
    "text, place",
    [
        ("I live in Toronto", "Toronto"),
        ("i moved to berlin last year", "Berlin"),
        ("I'm based in San Francisco and love it", "San Francisco"),
    ],
)
def test_where_he_says_he_lives_is_heard(text, place):
    assert stated_home(text) == place


def test_i_love_toronto_is_not_where_he_lives():
    assert stated_home("I love Toronto") is None


def test_the_weather_uses_his_city_before_guessing(monkeypatch):
    from ai.plugins.plugin_system import WeatherPlugin

    remember_home("Toronto")
    looked_up = []
    plugin = WeatherPlugin()
    monkeypatch.setattr(plugin, "_detect_location_fallback", lambda: "Somewhere Else")
    monkeypatch.setattr(plugin, "_get_coordinates", lambda location: looked_up.append(location))

    plugin.execute("weather:current", "how's the weather?", {}, {})

    assert looked_up == ["Toronto"]


def test_saying_where_he_lives_is_kept():
    from ai.runtime.companion_runtime import CompanionRuntimeLoop

    CompanionRuntimeLoop._capture_home_location("by the way, I live in Lisbon")

    assert home_location() == "Lisbon"


def test_the_answer_to_what_city_gets_the_weather_there():
    from ai.runtime.alice_contract_factory import build_runtime_boundaries
    from ai.runtime.contract_pipeline import ContractPipeline
    from tests.integration.test_contract_pipeline import _FakeAlice

    alice = _FakeAlice()
    alice._awaiting_weather_city = time.monotonic()

    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn(
        user_input="Toronto", user_id="u1", turn_number=2
    )

    assert result.metadata["intent"] == "weather:current"
    assert home_location() == "Toronto"
