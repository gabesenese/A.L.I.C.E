"""Safe background companion loop for desktop-agent behavior.

The daemon observes existing runtime state and proposes lightweight companion
actions. It does not execute tools or mutate user files; any output is limited
to notifications, world-state snapshots, and execution-journal audit entries.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, is_dataclass
from datetime import datetime
from enum import Enum
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CompanionDaemonConfig:
    interval_seconds: float = 60.0
    start_paused: bool = False
    notification_cooldown_seconds: float = 1800.0
    stale_goal_seconds: float = 4 * 60 * 60
    upcoming_reminder_window_seconds: float = 15 * 60


@dataclass
class CompanionDaemonDecision:
    decision_type: str
    key: str
    message: str
    reason: str
    priority: str = "normal"
    requires_approval: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "decision_type": self.decision_type,
            "key": self.key,
            "message": self.message,
            "reason": self.reason,
            "priority": self.priority,
            "requires_approval": self.requires_approval,
            "metadata": dict(self.metadata or {}),
        }


@dataclass
class CompanionObservation:
    captured_at: float
    system_snapshot: Dict[str, Any] = field(default_factory=dict)
    world_state: Dict[str, Any] = field(default_factory=dict)
    journal_summary: Dict[str, Any] = field(default_factory=dict)
    active_goals: List[Dict[str, Any]] = field(default_factory=list)
    pending_reminders: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "captured_at": self.captured_at,
            "system_snapshot": _json_safe(self.system_snapshot),
            "world_state": _json_safe(self.world_state),
            "journal_summary": _json_safe(self.journal_summary),
            "active_goals": _json_safe(self.active_goals),
            "pending_reminders": _json_safe(self.pending_reminders),
        }


class CompanionDaemon:
    """Observe ALICE's runtime and surface safe companion check-ins."""

    def __init__(
        self,
        *,
        state_api: Any = None,
        world_state_memory: Any = None,
        execution_journal: Any = None,
        goal_system: Any = None,
        proactive_assistant: Any = None,
        notify_callback: Optional[Callable[..., Any]] = None,
        config: CompanionDaemonConfig | None = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self.state_api = state_api
        self.world_state_memory = world_state_memory
        self.execution_journal = execution_journal
        self.goal_system = goal_system
        self.proactive_assistant = proactive_assistant
        self.notify_callback = notify_callback
        self.config = config or CompanionDaemonConfig()
        self._clock = clock or time.time

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._paused = bool(self.config.start_paused)
        self._tick_count = 0
        self._last_tick_at: float | None = None
        self._last_error = ""
        self._last_observation: Dict[str, Any] = {}
        self._last_decisions: List[Dict[str, Any]] = []
        self._last_notifications: Dict[str, float] = {}

    def start(self) -> None:
        """Start the daemon thread if it is not already running."""
        with self._lock:
            if self.is_running:
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._loop,
                name="alice-companion-daemon",
                daemon=True,
            )
            self._thread.start()
            logger.info("[CompanionDaemon] Started")

    def stop(self) -> None:
        """Stop the daemon thread."""
        thread = None
        with self._lock:
            self._stop_event.set()
            thread = self._thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2)
        with self._lock:
            self._thread = None
        logger.info("[CompanionDaemon] Stopped")

    def pause(self, reason: str = "manual") -> None:
        with self._lock:
            self._paused = True
            self._record_world_state(
                {
                    "paused": True,
                    "pause_reason": str(reason or "manual"),
                    "last_tick_at": self._last_tick_at,
                    "tick_count": self._tick_count,
                }
            )

    def resume(self) -> None:
        with self._lock:
            self._paused = False
            self._record_world_state(
                {
                    "paused": False,
                    "pause_reason": "",
                    "last_tick_at": self._last_tick_at,
                    "tick_count": self._tick_count,
                }
            )

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return bool(thread and thread.is_alive())

    @property
    def is_paused(self) -> bool:
        return bool(self._paused)

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "running": self.is_running,
                "paused": self.is_paused,
                "interval_seconds": float(self.config.interval_seconds),
                "tick_count": int(self._tick_count),
                "last_tick_at": self._last_tick_at,
                "last_error": self._last_error,
                "last_decisions": [dict(item) for item in self._last_decisions],
                "active_goal_count": len(
                    list((self._last_observation or {}).get("active_goals") or [])
                ),
                "pending_reminder_count": len(
                    list((self._last_observation or {}).get("pending_reminders") or [])
                ),
            }

    def run_once(self, *, reason: str = "manual") -> List[CompanionDaemonDecision]:
        """Run one observe-decide-persist cycle."""
        with self._lock:
            now = self._clock()
            self._tick_count += 1
            self._last_tick_at = now

            try:
                observation = self._observe(now=now)
                if self._paused:
                    decisions = [
                        CompanionDaemonDecision(
                            decision_type="noop",
                            key="companion:paused",
                            message="Companion daemon is paused.",
                            reason="paused",
                            metadata={"run_reason": reason},
                        )
                    ]
                else:
                    decisions = self._decide(observation)
                    if not decisions:
                        decisions = [
                            CompanionDaemonDecision(
                                decision_type="noop",
                                key="companion:idle",
                                message="No companion action needed.",
                                reason="no_actionable_signal",
                                metadata={"run_reason": reason},
                            )
                        ]

                self._emit_notifications(decisions, now=now)
                observation_dict = observation.as_dict()
                decision_dicts = [decision.as_dict() for decision in decisions]
                self._last_observation = observation_dict
                self._last_decisions = decision_dicts
                self._last_error = ""
                self._persist_cycle(
                    observation=observation_dict,
                    decisions=decision_dicts,
                    reason=reason,
                    now=now,
                )
                return decisions
            except Exception as exc:
                self._last_error = str(exc)
                logger.exception("[CompanionDaemon] Cycle failed: %s", exc)
                self._record_journal(
                    {
                        "event": "companion_daemon_error",
                        "success": False,
                        "error": str(exc),
                        "reason": reason,
                        "timestamp": now,
                    }
                )
                return [
                    CompanionDaemonDecision(
                        decision_type="noop",
                        key="companion:error",
                        message="Companion daemon cycle failed.",
                        reason="cycle_error",
                        priority="low",
                        metadata={"error": str(exc), "run_reason": reason},
                    )
                ]

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            if not self._paused:
                self.run_once(reason="scheduled")
            wait_seconds = max(1.0, float(self.config.interval_seconds or 60.0))
            self._stop_event.wait(wait_seconds)

    def _observe(self, *, now: float) -> CompanionObservation:
        system_snapshot = self._safe_call(self.state_api, "get_system_snapshot", {})
        world_state = self._safe_call(self.world_state_memory, "snapshot", {})
        journal_summary = self._safe_call(self.execution_journal, "summary", {})
        active_goals = self._collect_active_goals()
        pending_reminders = self._collect_pending_reminders(now=now)

        return CompanionObservation(
            captured_at=now,
            system_snapshot=dict(system_snapshot or {}),
            world_state=dict(world_state or {}),
            journal_summary=dict(journal_summary or {}),
            active_goals=active_goals,
            pending_reminders=pending_reminders,
        )

    def _decide(
        self, observation: CompanionObservation
    ) -> List[CompanionDaemonDecision]:
        decisions: List[CompanionDaemonDecision] = []
        now = float(observation.captured_at or self._clock())

        pending_approvals = list(
            (observation.world_state or {}).get("pending_approvals") or []
        )
        if pending_approvals:
            count = len(pending_approvals)
            decisions.append(
                CompanionDaemonDecision(
                    decision_type="notify",
                    key="companion:pending_approvals",
                    message=f"There are {count} pending approval(s) waiting.",
                    reason="pending_approval_queue",
                    priority="normal",
                    requires_approval=False,
                    metadata={"pending_approval_count": count},
                )
            )

        for goal in observation.active_goals[:5]:
            title = str(goal.get("title") or goal.get("goal_id") or "Untitled goal")
            goal_id = str(goal.get("goal_id") or title)
            blockers = list(goal.get("blockers") or [])
            if blockers:
                blocker_text = ", ".join(str(item) for item in blockers[:3])
                decisions.append(
                    CompanionDaemonDecision(
                        decision_type="ask",
                        key=f"companion:blocked_goal:{goal_id}",
                        message=(
                            f"Goal '{title}' is blocked: {blocker_text}. "
                            "I need direction before continuing."
                        ),
                        reason="goal_blocked",
                        priority="high",
                        requires_approval=False,
                        metadata={"goal_id": goal_id, "blockers": blockers[:5]},
                    )
                )
                continue

            updated_at = self._to_timestamp(goal.get("updated_at"))
            next_action = str(goal.get("next_action") or "").strip()
            if not next_action:
                next_action = str(
                    (goal.get("next_step") or {}).get("description") or ""
                ).strip()
            if updated_at and next_action:
                idle_seconds = max(0.0, now - updated_at)
                if idle_seconds >= float(self.config.stale_goal_seconds):
                    idle_hours = int(idle_seconds // 3600)
                    decisions.append(
                        CompanionDaemonDecision(
                            decision_type="ask",
                            key=f"companion:stale_goal:{goal_id}",
                            message=(
                                f"Goal '{title}' has been idle for {idle_hours}h. "
                                f"Next step: {next_action}"
                            ),
                            reason="goal_stale_with_next_action",
                            priority="normal",
                            metadata={
                                "goal_id": goal_id,
                                "idle_seconds": idle_seconds,
                                "next_action": next_action,
                            },
                        )
                    )

        health_messages = self._health_messages(observation.system_snapshot)
        for index, message in enumerate(health_messages[:3]):
            decisions.append(
                CompanionDaemonDecision(
                    decision_type="notify",
                    key=f"companion:health:{index}:{message[:80]}",
                    message=f"Runtime health needs attention: {message}",
                    reason="runtime_health_signal",
                    priority="normal",
                    metadata={"health_message": message},
                )
            )

        failures = int((observation.journal_summary or {}).get("failed", 0) or 0)
        successes = int((observation.journal_summary or {}).get("success", 0) or 0)
        if failures >= 3 and failures > successes:
            decisions.append(
                CompanionDaemonDecision(
                    decision_type="notify",
                    key="companion:journal:failures",
                    message=(
                        f"The recent execution journal shows {failures} failed "
                        "action(s). Review may be needed before more automation."
                    ),
                    reason="recent_execution_failures",
                    priority="normal",
                    metadata={"failed": failures, "success": successes},
                )
            )

        for reminder in observation.pending_reminders[:3]:
            trigger_ts = self._to_timestamp(reminder.get("trigger_time"))
            if not trigger_ts:
                continue
            if (
                now
                <= trigger_ts
                <= now + float(self.config.upcoming_reminder_window_seconds)
            ):
                message = str(reminder.get("message") or "Reminder")
                when = datetime.fromtimestamp(trigger_ts).strftime("%H:%M")
                decisions.append(
                    CompanionDaemonDecision(
                        decision_type="notify",
                        key=f"companion:reminder:{reminder.get('reminder_id') or message}",
                        message=f"Reminder coming up at {when}: {message}",
                        reason="upcoming_reminder",
                        priority=str(reminder.get("priority") or "normal"),
                        metadata=dict(reminder),
                    )
                )

        return decisions

    def _collect_active_goals(self) -> List[Dict[str, Any]]:
        goals = self._safe_call(self.goal_system, "get_active_goals", [])
        normalized: List[Dict[str, Any]] = []
        for goal in list(goals or [])[:20]:
            normalized.append(self._normalize_goal(goal))
        return normalized

    def _normalize_goal(self, goal: Any) -> Dict[str, Any]:
        if isinstance(goal, dict):
            data = dict(goal)
        else:
            status = getattr(goal, "status", "")
            priority = getattr(goal, "priority", "")
            data = {
                "goal_id": getattr(goal, "goal_id", ""),
                "title": getattr(goal, "title", ""),
                "description": getattr(goal, "description", ""),
                "status": getattr(status, "value", status),
                "priority": getattr(priority, "value", priority),
                "progress": getattr(goal, "progress", 0.0),
                "created_at": getattr(goal, "created_at", None),
                "updated_at": getattr(goal, "updated_at", None),
                "deadline": getattr(goal, "deadline", None),
                "blockers": list(getattr(goal, "blockers", []) or []),
                "success_criteria": list(getattr(goal, "success_criteria", []) or []),
                "current_step": getattr(goal, "current_step", None),
                "next_action": getattr(goal, "next_action", ""),
            }
            next_step = None
            if hasattr(goal, "get_next_step"):
                try:
                    next_step = goal.get_next_step()
                except Exception:
                    next_step = None
            if next_step is not None:
                data["next_step"] = self._normalize_step(next_step)

        if not data.get("next_action") and isinstance(data.get("next_step"), dict):
            data["next_action"] = str(
                (data.get("next_step") or {}).get("description") or ""
            ).strip()
        return _json_safe(data)

    def _normalize_step(self, step: Any) -> Dict[str, Any]:
        if isinstance(step, dict):
            return _json_safe(dict(step))
        return _json_safe(
            {
                "step_id": getattr(step, "step_id", ""),
                "description": getattr(step, "description", ""),
                "status": getattr(step, "status", ""),
                "step_type": getattr(step, "step_type", ""),
            }
        )

    def _collect_pending_reminders(self, *, now: float) -> List[Dict[str, Any]]:
        reminders = self._safe_call(self.proactive_assistant, "list_reminders", [])
        normalized: List[Dict[str, Any]] = []
        for reminder in list(reminders or [])[:20]:
            if isinstance(reminder, dict):
                data = dict(reminder)
            else:
                data = {
                    "reminder_id": getattr(reminder, "reminder_id", ""),
                    "message": getattr(reminder, "message", ""),
                    "trigger_time": getattr(reminder, "trigger_time", None),
                    "priority": getattr(reminder, "priority", "normal"),
                    "context": getattr(reminder, "context", {}) or {},
                }
            trigger_ts = self._to_timestamp(data.get("trigger_time"))
            if trigger_ts and trigger_ts >= now:
                data["trigger_timestamp"] = trigger_ts
                normalized.append(_json_safe(data))
        normalized.sort(key=lambda item: float(item.get("trigger_timestamp") or 0.0))
        return normalized

    def _health_messages(self, snapshot: Dict[str, Any]) -> List[str]:
        messages: List[str] = []
        errors = dict((snapshot or {}).get("errors") or {})
        recent_errors = list(errors.get("recent_errors") or [])
        if recent_errors:
            messages.append(f"{len(recent_errors)} recent error(s)")
        last_error = str(errors.get("last_error") or "").strip()
        if last_error:
            messages.append(last_error[:160])

        plugins = dict((snapshot or {}).get("plugins") or {})
        failed_plugins = list(plugins.get("failed_plugins") or [])
        if failed_plugins:
            messages.append(
                f"failed plugins: {', '.join(map(str, failed_plugins[:3]))}"
            )
        return messages

    def _emit_notifications(
        self, decisions: List[CompanionDaemonDecision], *, now: float
    ) -> None:
        for decision in decisions:
            if decision.decision_type not in {"notify", "ask"}:
                continue
            key = decision.key or decision.message
            last_emit = float(self._last_notifications.get(key, 0.0) or 0.0)
            cooldown = float(self.config.notification_cooldown_seconds or 0.0)
            if last_emit and now - last_emit < cooldown:
                decision.metadata["suppressed_by_cooldown"] = True
                decision.metadata["emitted"] = False
                continue

            self._last_notifications[key] = now
            decision.metadata["emitted"] = True
            self._notify(decision.message, decision.priority)

    def _notify(self, message: str, priority: str) -> None:
        if not self.notify_callback:
            return
        try:
            self.notify_callback(message, priority=priority)
        except TypeError:
            self.notify_callback(message, priority)
        except Exception as exc:
            logger.debug("[CompanionDaemon] Notification callback failed: %s", exc)

    def _persist_cycle(
        self,
        *,
        observation: Dict[str, Any],
        decisions: List[Dict[str, Any]],
        reason: str,
        now: float,
    ) -> None:
        self._record_world_state(
            {
                "paused": self._paused,
                "last_tick_at": now,
                "tick_count": self._tick_count,
                "last_run_reason": reason,
                "last_decisions": decisions,
                "active_goal_count": len(list(observation.get("active_goals") or [])),
                "pending_reminder_count": len(
                    list(observation.get("pending_reminders") or [])
                ),
                "journal_summary": dict(observation.get("journal_summary") or {}),
            }
        )
        self._record_journal(
            {
                "event": "companion_daemon_tick",
                "success": True,
                "reason": reason,
                "paused": self._paused,
                "decision_count": len(decisions),
                "decisions": decisions,
                "observation": {
                    "active_goal_count": len(
                        list(observation.get("active_goals") or [])
                    ),
                    "pending_reminder_count": len(
                        list(observation.get("pending_reminders") or [])
                    ),
                    "journal_summary": dict(observation.get("journal_summary") or {}),
                },
                "timestamp": now,
            }
        )

    def _record_world_state(self, data: Dict[str, Any]) -> None:
        if not self.world_state_memory or not hasattr(
            self.world_state_memory, "set_environment_state"
        ):
            return
        try:
            self.world_state_memory.set_environment_state(
                "companion_daemon",
                _json_safe(data),
                captured_at=data.get("last_tick_at") or self._clock(),
            )
        except Exception as exc:
            logger.debug("[CompanionDaemon] World-state write failed: %s", exc)

    def _record_journal(self, entry: Dict[str, Any]) -> None:
        if not self.execution_journal or not hasattr(self.execution_journal, "record"):
            return
        try:
            self.execution_journal.record(_json_safe(entry))
        except Exception as exc:
            logger.debug("[CompanionDaemon] Journal write failed: %s", exc)

    @staticmethod
    def _safe_call(target: Any, method_name: str, default: Any) -> Any:
        if not target or not hasattr(target, method_name):
            return default
        try:
            return getattr(target, method_name)()
        except Exception:
            return default

    @staticmethod
    def _to_timestamp(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, datetime):
            try:
                return float(value.timestamp())
            except (OSError, OverflowError, ValueError):
                return None
        text = str(value or "").strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            pass
        try:
            return datetime.fromisoformat(text).timestamp()
        except ValueError:
            return None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)
