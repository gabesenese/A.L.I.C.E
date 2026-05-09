from __future__ import annotations

from features import welcome


def _assert_no_banned_phrases(text: str) -> None:
    low = text.lower()
    banned = (
        "how can i help",
        "how may i assist",
        "i'm here to help",
        "i am here to help",
        "anything you need",
        "whatever you need",
        "systems online",
        "systems steady",
        "quiet mode",
        "critical path",
        "open loops",
        "handoff",
        "deep work",
        "cry for help",
        "responsible people",
        "one useful thing. then we reassess",
        "minimal and high-value",
        "give me the target",
        "map the shortest path",
        "point me",
        "i will map",
        "i will propose",
        "execution plan",
        "keep it surgical",
    )
    for token in banned:
        assert token not in low


def test_late_night_never_uses_old_consultant_phrases():
    for _ in range(128):
        greeting = welcome.get_greeting("Gabriel", "late_night")
        low = greeting.lower()
        assert "minimal and high-value" not in low
        assert "give me the target" not in low
        assert "map the shortest path" not in low


def test_all_periods_generate_valid_greetings():
    for period in (
        "early_morning",
        "morning",
        "afternoon",
        "evening",
        "night",
        "late_night",
    ):
        greeting = welcome.get_greeting("Gabriel", period)
        assert "Gabriel" in greeting
        lines = [line.strip() for line in greeting.splitlines() if line.strip()]
        assert 1 <= len(lines) <= 3
        assert all(len(line) <= 100 for line in lines)
        assert welcome._is_valid_startup_greeting(greeting) is True
        _assert_no_banned_phrases(greeting)


def test_multiline_output_present_for_late_night():
    greeting = welcome.get_greeting("Gabriel", "late_night")
    assert "\n\n" in greeting


def test_good_examples_accepted():
    good = (
        "Morning, Gabriel.\n\nWe have a plan. Allegedly.",
        "Afternoon, Gabriel.\n\nStill time for a clean win.",
        "Evening, Gabriel.\n\nBack for round two?",
        "Night session, Gabriel.\n\nBold choice. Let's make it worth it.",
        "Welcome back, Gabriel.\n\nLet's make the tabs earn their keep.",
        "Late night, Gabriel.\n\nLet's keep this clever, not chaotic.",
    )
    for sample in good:
        assert welcome._is_valid_startup_greeting(sample) is True


def test_bad_examples_rejected():
    bad = (
        "Systems online.",
        "Systems steady.",
        "Quiet mode.",
        "Point me at the blocker and I will propose the next move.",
        "I'm here to help with anything you need.",
        "Ideal time to wrap open loops and prepare tomorrow's handoff.",
        "One useful thing. Then we reassess.",
        "This is either focus or a cry for help.",
        "The responsible people have logged off.",
    )
    for sample in bad:
        assert welcome._is_valid_startup_greeting(sample) is False


def test_non_repetition_before_reset(monkeypatch):
    monkeypatch.setattr(
        welcome,
        "_GREETING_COMPONENTS",
        {
            "morning": {
                "openers": ["Morning, {name}."],
                "witty_lines": ["Line A.", "Line B."],
                "productive_nudges": ["Nudge A.", "Nudge B."],
            }
        },
    )
    monkeypatch.setattr(welcome, "_USED_GREETING_SIGNATURES", {"morning": set()})
    monkeypatch.setattr(welcome.random, "random", lambda: 0.1)
    seen = {welcome.get_greeting("Gabriel", "morning") for _ in range(4)}
    assert len(seen) == 4


def test_full_welcome_sequence_centers_each_line(monkeypatch):
    printed: list[str] = []
    monkeypatch.setattr(welcome, "get_terminal_width", lambda: 40)
    monkeypatch.setattr(welcome, "welcome_message", lambda *args, **kwargs: None)
    monkeypatch.setattr(welcome, "display_startup_info", lambda: None)
    monkeypatch.setattr(welcome, "animated_loading", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        welcome,
        "get_greeting",
        lambda *args, **kwargs: "Evening, Gabriel.\n\nBack for round two?",
    )
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: printed.append(" ".join(str(a) for a in args)))

    welcome.full_welcome_sequence("Gabriel", show_animation=False)

    assert any("Evening, Gabriel.".center(40).strip() in line for line in printed)
    assert any("Back for round two?".center(40).strip() in line for line in printed)
