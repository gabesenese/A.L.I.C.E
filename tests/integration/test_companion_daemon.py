from datetime import datetime, timedelta
from types import SimpleNamespace

from ai.runtime.companion_daemon import (
    CompanionDaemon,
    CompanionDaemonConfig,
    CompanionDaemonDecision,
)


class _WorldState:
    def __init__(self, state=None):
        self.state = dict(state or {})
        self.environment_writes = []

    def snapshot(self):
        return dict(self.state)

    def set_environment_state(self, key, data, *, captured_at=None):
        self.environment_writes.append((key, data, captured_at))


class _Journal:
    def __init__(self, summary=None):
        self.records = []
        self._summary = dict(summary or {})

    def summary(self):
        return dict(self._summary)

    def record(self, entry):
        self.records.append(dict(entry))


class _StateAPI:
    def __init__(self, snapshot=None):
        self.snapshot = dict(snapshot or {})

    def get_system_snapshot(self):
        return dict(self.snapshot)


class _Goal:
    def __init__(
        self,
        *,
        goal_id="goal-1",
        title="Ship desktop companion",
        updated_at=0.0,
        blockers=None,
        next_step="Implement daemon status command",
    ):
        self.goal_id = goal_id
        self.title = title
        self.description = title
        self.status = SimpleNamespace(value="in_progress")
        self.priority = SimpleNamespace(value=2)
        self.progress = 0.25
        self.created_at = 0.0
        self.updated_at = updated_at
        self.deadline = None
        self.blockers = list(blockers or [])
        self.success_criteria = ["daemon is observable"]
        self.current_step = None
        self._next_step = next_step

    def get_next_step(self):
        if not self._next_step:
            return None
        return SimpleNamespace(
            step_id="step-1",
            description=self._next_step,
            status="pending",
            step_type="implement",
        )


class _GoalSystem:
    def __init__(self, goals):
        self.goals = list(goals)

    def get_active_goals(self):
        return list(self.goals)


class _ProactiveAssistant:
    def __init__(self, reminders=None):
        self.reminders = list(reminders or [])

    def list_reminders(self):
        return list(self.reminders)


def test_companion_daemon_records_cycle_and_notifies_for_stale_goal():
    now = 10_000.0
    notifications = []
    world = _WorldState()
    journal = _Journal(summary={"success": 1, "failed": 0})
    goal_system = _GoalSystem(
        [_Goal(updated_at=now - 7200, next_step="Run the companion tests")]
    )
    daemon = CompanionDaemon(
        state_api=_StateAPI(),
        world_state_memory=world,
        execution_journal=journal,
        goal_system=goal_system,
        proactive_assistant=_ProactiveAssistant(),
        notify_callback=lambda message, priority="normal": notifications.append(
            (message, priority)
        ),
        config=CompanionDaemonConfig(
            stale_goal_seconds=3600,
            notification_cooldown_seconds=600,
        ),
        clock=lambda: now,
    )

    decisions = daemon.run_once(reason="test")

    assert decisions[0].decision_type == "ask"
    assert decisions[0].reason == "goal_stale_with_next_action"
    assert notifications == [
        ("Goal 'Ship desktop companion' has been idle for 2h. Next step: Run the companion tests", "normal")
    ]
    assert world.environment_writes[-1][0] == "companion_daemon"
    assert world.environment_writes[-1][1]["active_goal_count"] == 1
    assert journal.records[-1]["event"] == "companion_daemon_tick"
    assert journal.records[-1]["decision_count"] == 1


def test_companion_daemon_suppresses_repeated_notifications_during_cooldown():
    now = [10_000.0]
    notifications = []
    daemon = CompanionDaemon(
        state_api=_StateAPI(),
        world_state_memory=_WorldState(),
        execution_journal=_Journal(),
        goal_system=_GoalSystem([_Goal(updated_at=1000.0)]),
        proactive_assistant=_ProactiveAssistant(),
        notify_callback=lambda message, priority="normal": notifications.append(
            (message, priority)
        ),
        config=CompanionDaemonConfig(
            stale_goal_seconds=60,
            notification_cooldown_seconds=600,
        ),
        clock=lambda: now[0],
    )

    first = daemon.run_once(reason="first")
    second = daemon.run_once(reason="second")

    assert first[0].metadata["emitted"] is True
    assert second[0].metadata["suppressed_by_cooldown"] is True
    assert second[0].metadata["emitted"] is False
    assert len(notifications) == 1


def test_companion_daemon_prioritizes_blocked_goals():
    notifications = []
    daemon = CompanionDaemon(
        state_api=_StateAPI(),
        world_state_memory=_WorldState(),
        execution_journal=_Journal(),
        goal_system=_GoalSystem([_Goal(blockers=["needs user approval"])]),
        proactive_assistant=_ProactiveAssistant(),
        notify_callback=lambda message, priority="normal": notifications.append(
            (message, priority)
        ),
        clock=lambda: 10_000.0,
    )

    decisions = daemon.run_once(reason="test")

    assert decisions[0].decision_type == "ask"
    assert decisions[0].reason == "goal_blocked"
    assert decisions[0].requires_approval is False
    assert notifications[0][1] == "high"
    assert "needs user approval" in notifications[0][0]


def test_companion_daemon_observes_pending_approvals_and_reminders():
    now = 2_000_000_000.0
    reminder = SimpleNamespace(
        reminder_id="rem-1",
        message="standup",
        trigger_time=datetime.fromtimestamp(now) + timedelta(minutes=5),
        priority="normal",
        context={},
    )
    daemon = CompanionDaemon(
        state_api=_StateAPI(),
        world_state_memory=_WorldState(state={"pending_approvals": [{"id": "a1"}]}),
        execution_journal=_Journal(),
        goal_system=_GoalSystem([]),
        proactive_assistant=_ProactiveAssistant([reminder]),
        notify_callback=lambda *args, **kwargs: None,
        config=CompanionDaemonConfig(upcoming_reminder_window_seconds=900),
        clock=lambda: now,
    )

    decisions = daemon.run_once(reason="test")
    reasons = {decision.reason for decision in decisions}

    assert "pending_approval_queue" in reasons
    assert "upcoming_reminder" in reasons
    assert daemon.status()["pending_reminder_count"] == 1


def test_companion_daemon_paused_cycle_is_noop_but_audited():
    journal = _Journal()
    world = _WorldState()
    daemon = CompanionDaemon(
        state_api=_StateAPI(),
        world_state_memory=world,
        execution_journal=journal,
        goal_system=_GoalSystem([_Goal(updated_at=0.0)]),
        proactive_assistant=_ProactiveAssistant(),
        config=CompanionDaemonConfig(start_paused=True),
        clock=lambda: 10_000.0,
    )

    decisions = daemon.run_once(reason="manual")

    assert decisions[0].decision_type == "noop"
    assert decisions[0].reason == "paused"
    assert journal.records[-1]["paused"] is True
    assert world.environment_writes[-1][1]["paused"] is True


def test_alice_companion_command_controls_daemon(capsys):
    from app.main import ALICE

    class _Daemon:
        def __init__(self):
            self.running = False
            self.paused = False
            self.pause_reasons = []
            self.ticks = 0

        @property
        def is_running(self):
            return self.running

        def start(self):
            self.running = True

        def stop(self):
            self.running = False

        def pause(self, reason="manual"):
            self.paused = True
            self.pause_reasons.append(reason)

        def resume(self):
            self.paused = False

        def run_once(self, *, reason="manual"):
            self.ticks += 1
            return [
                CompanionDaemonDecision(
                    decision_type="noop",
                    key="test",
                    message="No companion action needed.",
                    reason=reason,
                )
            ]

        def status(self):
            return {
                "running": self.running,
                "paused": self.paused,
                "interval_seconds": 60,
                "tick_count": self.ticks,
                "last_tick_at": None,
                "last_error": "",
                "last_decisions": [],
                "active_goal_count": 0,
                "pending_reminder_count": 0,
            }

    alice = ALICE.__new__(ALICE)
    alice.companion_daemon = _Daemon()

    alice._handle_companion_command("/companion pause")
    alice._handle_companion_command("/companion resume")
    alice._handle_companion_command("/companion tick")

    output = capsys.readouterr().out
    assert alice.companion_daemon.pause_reasons == ["manual_command"]
    assert alice.companion_daemon.running is True
    assert alice.companion_daemon.paused is False
    assert alice.companion_daemon.ticks == 1
    assert "Companion Tick" in output
