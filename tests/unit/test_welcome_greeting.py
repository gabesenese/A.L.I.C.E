from __future__ import annotations

import json

from features import welcome
from features.welcome_validation import (
    is_valid_startup_greeting,
    validate_startup_greeting,
)


def test_validator_rejects_bad_assistant_and_consultant_greetings():
    bad_1 = (
        "Still online, Gabriel. I can help you finish one important task and park the rest. "
        "Want a quick status sweep and a concrete plan?"
    )
    bad_2 = (
        "Quiet hours, Gabriel. Let's keep it minimal and high-value. "
        "Give me the target and I'll map the shortest path."
    )
    assert is_valid_startup_greeting(bad_1) is False
    assert is_valid_startup_greeting(bad_2) is False


def test_validator_accepts_good_curated_greetings():
    assert is_valid_startup_greeting("Morning, Gabriel.\n\nWe have a plan. Allegedly.")
    assert is_valid_startup_greeting(
        "Late night, Gabriel.\n\nLet's keep this clever, not chaotic."
    )


def test_get_greeting_does_not_call_network(monkeypatch):
    def _explode(*_args, **_kwargs):
        raise AssertionError("network should not be called by get_greeting")

    monkeypatch.setattr("urllib.request.urlopen", _explode, raising=False)
    out = welcome.get_greeting("Gabriel", "late_night")
    assert "Gabriel" in out
    assert is_valid_startup_greeting(out)


def test_approved_greetings_load_and_format(monkeypatch, tmp_path):
    approved = {
        "early_morning": [],
        "morning": [],
        "afternoon": [],
        "evening": [],
        "night": [],
        "late_night": ["Late night, {name}.\n\nLet's keep this clever, not chaotic."],
    }
    p = tmp_path / "welcome_greetings.approved.json"
    p.write_text(json.dumps(approved), encoding="utf-8")
    monkeypatch.setattr(welcome, "_APPROVED_GREETINGS_PATH", p)
    out = welcome.get_greeting("Gabriel", "late_night")
    assert out == "Late night, Gabriel.\n\nLet's keep this clever, not chaotic."


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
        assert all(len(line) <= 90 for line in lines)
        assert is_valid_startup_greeting(greeting)


def test_non_repetition_before_reset(monkeypatch):
    monkeypatch.setattr(
        welcome,
        "_GREETING_MESSAGES",
        {
            "morning": [
                "Morning, {name}.\n\nLine A.",
                "Morning, {name}.\n\nLine B.",
                "Morning, {name}.\n\nLine C.",
            ]
        },
    )
    monkeypatch.setattr(welcome, "_USED_GREETING_SIGNATURES", {"morning": set()})
    monkeypatch.setattr(welcome, "_load_approved_greetings", lambda: {})
    seen = {welcome.get_greeting("Gabriel", "morning") for _ in range(3)}
    assert len(seen) == 3


def test_full_welcome_sequence_centers_each_line(monkeypatch):
    printed: list[str] = []
    monkeypatch.setattr(welcome, "get_terminal_width", lambda: 40)
    monkeypatch.setattr(welcome, "welcome_message", lambda *args, **kwargs: None)
    monkeypatch.setattr(welcome, "display_startup_info", lambda: None)
    monkeypatch.setattr(welcome, "animated_loading", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        welcome,
        "get_greeting",
        lambda *args, **kwargs: "Evening, Gabriel.\n\nBack for round two.",
    )
    monkeypatch.setattr(
        "builtins.print",
        lambda *args, **kwargs: printed.append(" ".join(str(a) for a in args)),
    )
    welcome.full_welcome_sequence("Gabriel", show_animation=False)
    assert any("Evening, Gabriel.".center(40).strip() in line for line in printed)
    assert any("Back for round two.".center(40).strip() in line for line in printed)


def test_validator_reasons_include_question_mark():
    reasons = validate_startup_greeting("Morning, {name}?\n\nWe have a plan.")
    assert "contains_question_mark" in reasons
