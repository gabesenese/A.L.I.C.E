"""Tests for AmbientMonitor — verifies world model writes without real API calls."""

from datetime import datetime, timezone

from brain.ambient_monitor import AmbientMonitor, AmbientConfig
from memory.world_model import WorldModel


def _clock():
    return datetime.now(timezone.utc)


def test_ambient_monitor_poll_goals_writes_stale_task_to_world_model(tmp_path, monkeypatch):
    from ai.goals.goal_engine import GoalEngine

    engine = GoalEngine(goals_file=tmp_path / "goals.json")
    engine.add("finish the routing refactor", priority=1)
    # Mark last_worked_at as old enough to be stale
    for goal in engine.active():
        engine.update(goal.goal_id, last_worked_at="2020-01-01T00:00:00+00:00")

    monkeypatch.setattr(
        "brain.ambient_monitor.AmbientMonitor._poll_calendar",
        lambda self: {"skipped": "test"},
    )
    monkeypatch.setattr(
        "brain.ambient_monitor.AmbientMonitor._poll_email",
        lambda self: {"skipped": "test"},
    )

    # Patch get_goal_engine to return our controlled engine
    import ai.goals.goal_engine as ge_mod

    monkeypatch.setattr(ge_mod, "get_goal_engine", lambda: engine)

    model = WorldModel(tmp_path / "world_model.json")
    monitor = AmbientMonitor(world_model=model, config=AmbientConfig(stale_goal_days=1.0))
    result = monitor.run_once()

    assert result["goals"].get("stale_count", 0) >= 1
    snapshot = model.snapshot()
    open_texts = [t["text"] for t in snapshot["environment"]["open_tasks"]]
    assert any("routing refactor" in t.lower() for t in open_texts)


def test_ambient_monitor_update_upcoming_calendar(tmp_path):
    model = WorldModel(tmp_path / "world_model.json")
    events = [
        {
            "title": "Stand-up",
            "start": "2026-05-18T09:00:00+00:00",
            "end": "2026-05-18T09:30:00+00:00",
            "location": "",
            "all_day": False,
        },
        {
            "title": "Review",
            "start": "2026-05-18T14:00:00+00:00",
            "end": "2026-05-18T15:00:00+00:00",
            "location": "",
            "all_day": False,
        },
    ]
    model.update_upcoming_calendar(events)
    snapshot = model.snapshot()
    assert len(snapshot["environment"]["upcoming_calendar"]) == 2
    assert snapshot["environment"]["upcoming_calendar"][0]["title"] == "Stand-up"


def test_ambient_monitor_update_unread_email_count(tmp_path):
    model = WorldModel(tmp_path / "world_model.json")
    model.update_unread_email_count(7)
    assert model.snapshot()["environment"]["unread_email_count"] == 7
    model.update_unread_email_count(0)
    assert model.snapshot()["environment"]["unread_email_count"] == 0


def test_add_open_task_if_missing_deduplicates(tmp_path):
    model = WorldModel(tmp_path / "world_model.json")
    added1 = model.add_open_task_if_missing(text="Fix the routing bug", source="test")
    added2 = model.add_open_task_if_missing(text="Fix the routing bug", source="test")
    assert added1 is True
    assert added2 is False
    tasks = model.snapshot()["environment"]["open_tasks"]
    assert len(tasks) == 1
