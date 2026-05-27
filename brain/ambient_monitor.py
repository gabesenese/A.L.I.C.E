"""Background ambient monitor — polls real data sources and feeds the world model.

Runs every 5 minutes by default. Never writes to stdout. All sources degrade
gracefully: if a plugin is unauthenticated or unavailable the poll is skipped
silently so a missing Google token never crashes startup.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

from memory.world_model import WorldModel, get_world_model

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AmbientConfig:
    interval_seconds: float = 300.0  # poll every 5 minutes
    warmup_seconds: float = 30.0  # delay before first tick so startup isn't blocked
    calendar_lookahead_hours: float = 24.0
    stale_goal_days: float = 3.0


class AmbientMonitor:
    """Polls calendar, email, and goal engine; writes findings into the world model.

    The heartbeat reads the world model — this feeds it with real data so the
    heartbeat can fire on actual events rather than only on conversation signals.
    """

    def __init__(
        self,
        *,
        world_model: WorldModel | None = None,
        config: AmbientConfig | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.world_model = world_model or get_world_model()
        self.config = config or AmbientConfig()
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()

    @property
    def is_running(self) -> bool:
        t = self._thread
        return bool(t and t.is_alive())

    def start(self) -> None:
        with self._lock:
            if self.is_running:
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._loop,
                name="alice-ambient",
                daemon=True,
            )
            self._thread.start()
            logger.info(
                "[AmbientMonitor] started (interval=%.0fs)",
                self.config.interval_seconds,
            )

    def stop(self) -> None:
        thread = None
        with self._lock:
            self._stop_event.set()
            thread = self._thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=3)
        with self._lock:
            self._thread = None

    def run_once(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        results["calendar"] = self._poll_calendar()
        results["email"] = self._poll_email()
        results["goals"] = self._poll_goals()
        results["system"] = self._poll_system()
        return results

    # ------------------------------------------------------------------ loop

    def _loop(self) -> None:
        self._stop_event.wait(max(1.0, float(self.config.warmup_seconds)))
        while not self._stop_event.is_set():
            try:
                self.run_once()
            except Exception as exc:
                logger.debug("[AmbientMonitor] tick error: %s", exc)
            self._stop_event.wait(max(60.0, float(self.config.interval_seconds)))

    # ------------------------------------------------------------------ polls

    def _poll_calendar(self) -> Dict[str, Any]:
        try:
            from ai.plugins.calendar_plugin import CalendarPlugin, GOOGLE_AVAILABLE  # noqa: PLC0415

            if not GOOGLE_AVAILABLE:
                return {"skipped": "google_unavailable"}

            plugin = CalendarPlugin()
            if not getattr(plugin, "service", None):
                return {"skipped": "not_authenticated"}

            now = self.clock()
            end = now + timedelta(hours=self.config.calendar_lookahead_hours)

            events_result = (
                plugin.service.events()
                .list(
                    calendarId="primary",
                    timeMin=now.isoformat(),
                    timeMax=end.isoformat(),
                    maxResults=15,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )
            items = events_result.get("items", [])
            events: List[Dict[str, Any]] = []
            for item in items:
                start_raw = item["start"].get("dateTime", item["start"].get("date", ""))
                end_raw = item["end"].get("dateTime", item["end"].get("date", ""))
                events.append(
                    {
                        "title": str(item.get("summary") or "Untitled"),
                        "start": start_raw,
                        "end": end_raw,
                        "location": str(item.get("location") or ""),
                        "all_day": "T" not in start_raw,
                    }
                )

            self.world_model.update_upcoming_calendar(events)
            logger.debug("[AmbientMonitor] calendar: %d events", len(events))
            return {"count": len(events)}
        except Exception as exc:
            logger.debug("[AmbientMonitor] calendar poll skipped: %s", exc)
            return {"skipped": str(exc)}

    def _poll_email(self) -> Dict[str, Any]:
        try:
            from ai.plugins.email_plugin import GmailPlugin, GMAIL_AVAILABLE  # noqa: PLC0415

            if not GMAIL_AVAILABLE:
                return {"skipped": "gmail_unavailable"}

            plugin = GmailPlugin()
            if not getattr(plugin, "service", None):
                return {"skipped": "not_authenticated"}

            count = plugin.get_unread_count()
            self.world_model.update_unread_email_count(count)
            logger.debug("[AmbientMonitor] email: %d unread", count)
            return {"unread": count}
        except Exception as exc:
            logger.debug("[AmbientMonitor] email poll skipped: %s", exc)
            return {"skipped": str(exc)}

    def _poll_goals(self) -> Dict[str, Any]:
        try:
            from ai.goals.goal_engine import get_goal_engine  # noqa: PLC0415

            engine = get_goal_engine()
            stale = engine.stale_goals(days=self.config.stale_goal_days)
            if stale:
                now = self.clock().isoformat()
                for goal in stale:
                    self.world_model.add_open_task_if_missing(
                        text=f"Goal stalled: {goal.description}",
                        source="ambient_goal_staleness",
                        timestamp=now,
                    )
            logger.debug("[AmbientMonitor] goals: %d stale", len(stale))
            return {"stale_count": len(stale)}
        except Exception as exc:
            logger.debug("[AmbientMonitor] goal poll skipped: %s", exc)
            return {"skipped": str(exc)}

    def _poll_system(self) -> Dict[str, Any]:
        try:
            from ai.plugins.system_plugin import SystemPlugin  # noqa: PLC0415

            snap = SystemPlugin().get_snapshot()
            if snap:
                self.world_model._state["environment"]["system_health"] = snap
                self.world_model.save()
            logger.debug("[AmbientMonitor] system: cpu=%.0f%%", snap.get("cpu_percent", 0))
            return snap
        except Exception as exc:
            logger.debug("[AmbientMonitor] system poll skipped: %s", exc)
            return {"skipped": str(exc)}


_monitor: Optional[AmbientMonitor] = None


def get_ambient_monitor() -> AmbientMonitor:
    global _monitor
    if _monitor is None:
        _monitor = AmbientMonitor()
    return _monitor
