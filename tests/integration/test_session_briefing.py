"""Tests for the expanded session briefing and companion_runtime session continuity."""

from datetime import datetime, timezone
from unittest.mock import patch

from memory.world_model import WorldModel
from ai.runtime.session_briefing import generate_session_briefing


def test_briefing_does_not_include_email_count(tmp_path, monkeypatch):
    """Email count is intentionally excluded from the session briefing.
    Users should ask ALICE directly for email status."""
    model = WorldModel(tmp_path / "world_model.json")
    model.update_unread_email_count(5)
    monkeypatch.setattr("memory.world_model._world_model", model)

    with (
        patch("ai.identity.user_identity.load_identity") as mock_id,
        patch("ai.goals.goal_engine.get_goal_engine") as mock_ge,
        patch("memory.world_model.get_world_model", return_value=model),
    ):
        mock_id.return_value = type("I", (), {"name": "Gabriel"})()
        mock_engine = type(
            "E",
            (),
            {
                "session_summary": lambda self: "",
                "stale_goals": lambda self, days=3.0: [],
                "active": lambda self: [],
            },
        )()
        mock_ge.return_value = mock_engine

        briefing = generate_session_briefing("gabriel")

    assert "5 unread" not in briefing
    assert "unread" not in briefing


def test_briefing_includes_calendar_events(tmp_path):
    model = WorldModel(tmp_path / "world_model.json")
    model.update_upcoming_calendar(
        [
            {
                "title": "Sprint planning",
                "start": datetime.now(timezone.utc)
                .replace(hour=10, minute=0)
                .isoformat(),
                "end": datetime.now(timezone.utc)
                .replace(hour=11, minute=0)
                .isoformat(),
                "location": "",
                "all_day": "False",
            }
        ]
    )

    with (
        patch("ai.identity.user_identity.load_identity") as mock_id,
        patch("ai.goals.goal_engine.get_goal_engine") as mock_ge,
        patch("memory.world_model.get_world_model", return_value=model),
    ):
        mock_id.return_value = type("I", (), {"name": "Gabriel"})()
        mock_engine = type(
            "E",
            (),
            {
                "session_summary": lambda self: "",
                "stale_goals": lambda self, days=3.0: [],
            },
        )()
        mock_ge.return_value = mock_engine

        briefing = generate_session_briefing("gabriel")

    assert "Sprint planning" in briefing


def test_session_continuity_injected_on_first_turn(tmp_path):
    from ai.runtime.companion_runtime import CompanionRuntimeLoop

    model = WorldModel(tmp_path / "world_model.json")
    model.update_from_turn(
        user_input="I want to finish the ambient monitor. I need to write tests.",
        response_text="On it.",
        timestamp="2026-05-17T10:00:00+00:00",
    )

    loop = CompanionRuntimeLoop(world_model=model)
    state = loop.start_turn(
        user_id="gabriel",
        user_input="Hey",
        turn_number=1,
        user_state=None,
    )

    identity = state.memory_domains.identity
    continuity = identity.get("session_continuity") or {}
    assert (
        "open_tasks" in continuity
        or "active_goals" in continuity
        or "active_threads" in continuity
    )


def test_session_continuity_not_reinjected_on_subsequent_turns(tmp_path):
    from ai.runtime.companion_runtime import CompanionRuntimeLoop

    model = WorldModel(tmp_path / "world_model.json")
    model.update_from_turn(
        user_input="I need to review the heartbeat.",
        response_text="Noted.",
        timestamp="2026-05-17T10:00:00+00:00",
    )

    loop = CompanionRuntimeLoop(world_model=model)

    # First turn — session is new, continuity injected
    state1 = loop.start_turn(
        user_id="gabriel", user_input="Hey", turn_number=1, user_state=None
    )
    assert "session_continuity" in (state1.memory_domains.identity or {})

    # Later turn — session already exists, continuity NOT re-injected
    state2 = loop.start_turn(
        user_id="gabriel", user_input="What next?", turn_number=5, user_state=None
    )
    assert "session_continuity" not in (state2.memory_domains.identity or {})
