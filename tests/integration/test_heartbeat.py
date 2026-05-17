from datetime import datetime, timezone

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
