"""Tests for TaskScheduler — verifies scheduling logic and callback firing."""

from datetime import datetime


from brain.task_scheduler import ScheduledTask, TaskScheduler


def _make_scheduler(tmp_path):
    return TaskScheduler(tasks_file=tmp_path / "tasks.json")


def test_daily_task_fires_at_correct_hour(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    task = ScheduledTask(name="test_daily", action="hello", schedule="daily@09:00")
    scheduler.add_task(task)

    due_time = datetime(2026, 5, 18, 9, 0, 30)
    not_due_time = datetime(2026, 5, 18, 10, 0, 0)

    assert task.is_due(due_time) is True
    assert task.is_due(not_due_time) is False


def test_daily_task_does_not_fire_twice_same_day(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    task = ScheduledTask(
        name="test_once",
        action="briefing",
        schedule="daily@09:00",
        last_run="2026-05-18T09:00:00+00:00",
    )
    scheduler.add_task(task)
    due_time = datetime(2026, 5, 18, 9, 0, 30)
    assert task.is_due(due_time) is False


def test_interval_task_fires_after_elapsed_time(tmp_path):
    task = ScheduledTask(
        name="test_interval",
        action="check",
        schedule="interval@30min",
        last_run="2026-05-18T08:00:00+00:00",
    )
    due = datetime(2026, 5, 18, 8, 31, 0)
    not_due = datetime(2026, 5, 18, 8, 15, 0)
    assert task.is_due(due) is True
    assert task.is_due(not_due) is False


def test_weekday_task_only_fires_on_correct_day(tmp_path):
    task = ScheduledTask(
        name="test_weekday",
        action="weekly_review",
        schedule="weekday@MON@09:00",
    )
    monday = datetime(2026, 5, 18, 9, 0)  # 2026-05-18 is a Monday
    tuesday = datetime(2026, 5, 19, 9, 0)
    assert task.is_due(monday) is True
    assert task.is_due(tuesday) is False


def test_callback_is_invoked_when_task_fires(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    fired: list[str] = []

    scheduler.register_callback(lambda name, action: fired.append(action) or "ok")
    task = ScheduledTask(
        name="immediate",
        action="trigger_me",
        schedule="interval@1min",
    )
    scheduler.add_task(task)
    scheduler._fire(task)

    assert "trigger_me" in fired


def test_default_daily_briefing_task_seeded_as_disabled(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    task = scheduler._tasks.get("daily_briefing")
    assert task is not None
    assert task.enabled is False
    assert "09:00" in task.schedule


def test_task_persists_across_reload(tmp_path):
    path = tmp_path / "tasks.json"
    s1 = TaskScheduler(tasks_file=path)
    s1.add_task(ScheduledTask(name="persist_me", action="wake_up", schedule="daily@07:00"))

    s2 = TaskScheduler(tasks_file=path)
    assert "persist_me" in s2._tasks
