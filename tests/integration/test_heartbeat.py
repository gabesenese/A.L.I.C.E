from datetime import datetime, timedelta, timezone

from brain.heartbeat import Heartbeat, HeartbeatConfig
from memory.world_model import WorldModel


class _Clock:
    def __init__(self, iso: str):
        self.value = datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)

    def __call__(self):
        return self.value


def test_heartbeat_interrupts_for_stale_intention_once_until_resolved(tmp_path):
    model = WorldModel(tmp_path / "world_model.json")
    model.update_from_turn(
        user_input="I want to review the routing policy.",
        response_text="Tracked.",
        timestamp="2026-05-17T08:00:00+00:00",
    )
    output = []
    clock = _Clock("2026-05-17T11:01:00+00:00")
    heartbeat = Heartbeat(
        world_model=model,
        config=HeartbeatConfig(min_interrupt_gap_seconds=0),
        output=output.append,
        clock=clock,
    )

    first = heartbeat.run_once()
    second = heartbeat.run_once()

    assert len(first) == 1
    assert first[0].reason == "stale_intention"
    assert "A.L.I.C.E:" in output[0]
    assert second == []

    model.mark_thread_resolved("review the routing policy")
    assert model.snapshot()["user"]["mentioned_intentions"] == []


def test_heartbeat_general_checkin_updates_last_checkin(tmp_path):
    model = WorldModel(tmp_path / "world_model.json")
    model.update_from_turn(
        user_input="I am working on Alice.",
        response_text="Understood.",
        timestamp="2026-05-17T10:00:00+00:00",
    )
    output = []
    heartbeat = Heartbeat(
        world_model=model,
        config=HeartbeatConfig(min_interrupt_gap_seconds=0),
        output=output.append,
        clock=_Clock("2026-05-17T11:40:00+00:00"),
    )

    decisions = heartbeat.run_once()

    assert len(decisions) == 1
    assert decisions[0].reason == "general_checkin_due"
    assert model.snapshot()["user"]["last_checkin"]


def test_heartbeat_surfaces_open_tasks_older_than_24_hours(tmp_path):
    model = WorldModel(tmp_path / "world_model.json")
    model.update_from_turn(
        user_input="I should clean up the memory model.",
        response_text="Tracked.",
        timestamp="2026-05-16T07:00:00+00:00",
    )
    output = []
    heartbeat = Heartbeat(
        world_model=model,
        config=HeartbeatConfig(min_interrupt_gap_seconds=0),
        output=output.append,
        clock=_Clock("2026-05-17T08:01:00+00:00"),
    )

    decisions = heartbeat.run_once()

    assert len(decisions) == 1
    assert decisions[0].reason == "stale_open_task"

    model.mark_thread_resolved("clean up the memory model")
    model.update_from_turn(
        user_input="task: check old calendar placeholder",
        response_text="Tracked.",
        timestamp="2026-05-16T07:00:00+00:00",
    )
    heartbeat = Heartbeat(
        world_model=model,
        config=HeartbeatConfig(min_interrupt_gap_seconds=0),
        output=output.append,
        clock=_Clock("2026-05-17T08:01:00+00:00"),
    )
    task_decisions = heartbeat.run_once()

    assert len(task_decisions) == 1
    assert task_decisions[0].reason == "stale_open_task"


def test_heartbeat_alerts_for_imminent_meeting(tmp_path):
    model = WorldModel(tmp_path / "world_model.json")
    now = datetime.now(timezone.utc)
    in_ten_minutes = (now + timedelta(minutes=10)).isoformat()
    model.update_upcoming_calendar(
        [
            {
                "title": "Sprint planning",
                "start": in_ten_minutes,
                "end": in_ten_minutes,
                "location": "",
                "all_day": "False",
            },
        ]
    )

    output = []
    heartbeat = Heartbeat(
        world_model=model,
        config=HeartbeatConfig(min_interrupt_gap_seconds=0),
        output=output.append,
    )

    decisions = heartbeat.run_once()

    assert any(d.reason == "imminent_meeting" for d in decisions)
    assert any("Sprint planning" in d.message for d in decisions)


def test_heartbeat_no_alert_for_all_day_events(tmp_path):
    model = WorldModel(tmp_path / "world_model.json")
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")
    model.update_upcoming_calendar(
        [
            {"title": "Holiday", "start": today, "end": today, "location": "", "all_day": "True"},
        ]
    )

    output = []
    heartbeat = Heartbeat(
        world_model=model,
        config=HeartbeatConfig(min_interrupt_gap_seconds=0),
        output=output.append,
    )

    decisions = heartbeat.run_once()
    assert not any(d.reason == "imminent_meeting" for d in decisions)


def test_heartbeat_alerts_for_critical_cpu(tmp_path):
    model = WorldModel(tmp_path / "world_model.json")
    model._state["environment"]["system_health"] = {
        "cpu_percent": 95.0,
        "ram_percent": 50.0,
        "ram_used_gb": 8.0,
        "ram_total_gb": 16.0,
    }
    model.save()

    output = []
    heartbeat = Heartbeat(
        world_model=model,
        config=HeartbeatConfig(min_interrupt_gap_seconds=0),
        output=output.append,
    )

    decisions = heartbeat.run_once()
    assert any(d.reason == "cpu_critical" for d in decisions)
    assert any("95" in d.message for d in decisions)


def test_heartbeat_alerts_for_critical_ram(tmp_path):
    model = WorldModel(tmp_path / "world_model.json")
    model._state["environment"]["system_health"] = {
        "cpu_percent": 20.0,
        "ram_percent": 92.0,
        "ram_used_gb": 14.7,
        "ram_total_gb": 16.0,
    }
    model.save()

    output = []
    heartbeat = Heartbeat(
        world_model=model,
        config=HeartbeatConfig(min_interrupt_gap_seconds=0),
        output=output.append,
    )

    decisions = heartbeat.run_once()
    assert any(d.reason == "ram_critical" for d in decisions)
