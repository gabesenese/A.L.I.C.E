from features import welcome


def test_get_greeting_includes_name_and_agentic_prompt():
    greeting = welcome.get_greeting(name="Gabriel", time_of_day="morning")

    assert "Gabriel" in greeting
    assert "I will" in greeting


def test_get_greeting_accepts_extended_time_alias():
    greeting = welcome.get_greeting(name="Gabriel", time_of_day="late night")

    assert "Gabriel" in greeting
    assert greeting.strip().endswith(".")


def test_get_greeting_uses_non_repeating_combos_before_reset(monkeypatch):
    monkeypatch.setattr(
        welcome,
        "_GREETING_COMPONENTS",
        {
            "morning": {
                "openers": ["Good morning, {name}."],
                "context": ["Context A.", "Context B."],
                "agentic_prompt": ["I will plan it."],
            }
        },
    )
    monkeypatch.setattr(welcome, "_USED_GREETING_SIGNATURES", {"morning": set()})

    first = welcome.get_greeting(name="Gabriel", time_of_day="morning")
    second = welcome.get_greeting(name="Gabriel", time_of_day="morning")

    assert first != second
