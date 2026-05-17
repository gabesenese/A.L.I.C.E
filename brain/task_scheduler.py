"""Scheduled autonomous task runner.

Tasks are stored in data/scheduled_tasks.json and executed by injecting the
task's action string directly into the contract pipeline via a registered
callback. This enables pre-approved recurring work (daily briefing, email
digest, etc.) without requiring Gabriel to ask every time.

Schedule formats:
  "daily@HH:MM"              — fires once per day at that local time
  "interval@Xmin"            — fires every X minutes
  "weekday@MON@HH:MM"        — fires on a specific weekday (MON-SUN) at HH:MM
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_TASKS_FILE = Path("data/scheduled_tasks.json")
_CHECK_INTERVAL = 60.0  # seconds between schedule checks

_WEEKDAY_MAP = {
    "MON": 0, "TUE": 1, "WED": 2, "THU": 3,
    "FRI": 4, "SAT": 5, "SUN": 6,
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _local_now() -> datetime:
    return datetime.now()


@dataclass
class ScheduledTask:
    name: str
    action: str           # injected as user input to the pipeline
    schedule: str         # "daily@HH:MM" | "interval@Xmin" | "weekday@DAY@HH:MM"
    enabled: bool = True
    last_run: str = ""    # ISO timestamp of last execution
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScheduledTask":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def is_due(self, now: Optional[datetime] = None) -> bool:
        now = now or _local_now()
        parts = str(self.schedule or "").split("@")
        kind = parts[0].lower()

        last = self._parse_last_run()

        if kind == "daily" and len(parts) == 2:
            target_hm = parts[1]
            try:
                h, m = map(int, target_hm.split(":"))
            except ValueError:
                return False
            if last and last.date() == now.date():
                return False
            return now.hour == h and now.minute == m

        if kind == "interval" and len(parts) == 2:
            try:
                minutes = int(parts[1].lower().replace("min", "").strip())
            except ValueError:
                return False
            if last is None:
                return True
            elapsed = (now - last.replace(tzinfo=None)).total_seconds()
            return elapsed >= minutes * 60

        if kind == "weekday" and len(parts) == 3:
            day_key = parts[1].upper()
            target_hm = parts[2]
            target_weekday = _WEEKDAY_MAP.get(day_key)
            if target_weekday is None:
                return False
            try:
                h, m = map(int, target_hm.split(":"))
            except ValueError:
                return False
            if now.weekday() != target_weekday:
                return False
            if last and last.date() == now.date():
                return False
            return now.hour == h and now.minute == m

        return False

    def _parse_last_run(self) -> Optional[datetime]:
        if not self.last_run:
            return None
        try:
            return datetime.fromisoformat(str(self.last_run).replace("Z", "+00:00"))
        except ValueError:
            return None


class TaskScheduler:
    """Runs scheduled tasks by firing a callback with the task's action string.

    Usage:
        scheduler = TaskScheduler()
        scheduler.register_callback(lambda action: alice.process_input(action))
        scheduler.start()
    """

    def __init__(self, tasks_file: Path | str = _TASKS_FILE) -> None:
        self._file = Path(tasks_file)
        self._file.parent.mkdir(parents=True, exist_ok=True)
        self._tasks: Dict[str, ScheduledTask] = {}
        self._callback: Optional[Callable[[str, str], Optional[str]]] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._load()
        self._seed_defaults()

    # ----------------------------------------------------------------- public

    def register_callback(self, callback: Callable[[str, str], Optional[str]]) -> None:
        """Register the function called when a task fires.

        Signature: callback(task_name: str, action: str) -> Optional[str]
        The return value (ALICE's response) is logged but not displayed unless
        the callback handles output itself.
        """
        self._callback = callback

    def add_task(self, task: ScheduledTask) -> None:
        with self._lock:
            self._tasks[task.name] = task
            self._save()

    def remove_task(self, name: str) -> bool:
        with self._lock:
            if name not in self._tasks:
                return False
            del self._tasks[name]
            self._save()
            return True

    def set_enabled(self, name: str, enabled: bool) -> bool:
        with self._lock:
            task = self._tasks.get(name)
            if task is None:
                return False
            task.enabled = enabled
            self._save()
            return True

    def list_tasks(self) -> List[ScheduledTask]:
        with self._lock:
            return list(self._tasks.values())

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
                name="alice-scheduler",
                daemon=True,
            )
            self._thread.start()
            logger.info("[TaskScheduler] started (%d tasks)", len(self._tasks))

    def stop(self) -> None:
        thread = None
        with self._lock:
            self._stop_event.set()
            thread = self._thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=3)
        with self._lock:
            self._thread = None

    # ----------------------------------------------------------------- loop

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self._tick()
            self._stop_event.wait(_CHECK_INTERVAL)

    def _tick(self) -> None:
        with self._lock:
            tasks = list(self._tasks.values())

        now = _local_now()
        for task in tasks:
            if not task.enabled:
                continue
            try:
                if task.is_due(now):
                    self._fire(task)
            except Exception as exc:
                logger.debug("[TaskScheduler] error checking task %s: %s", task.name, exc)

    def _fire(self, task: ScheduledTask) -> None:
        logger.info("[TaskScheduler] firing task: %s", task.name)
        with self._lock:
            task.last_run = _now().isoformat()
            self._save()

        if self._callback is None:
            logger.debug("[TaskScheduler] no callback registered, skipping execution")
            return
        try:
            result = self._callback(task.name, task.action)
            if result:
                logger.debug("[TaskScheduler] task %s response: %.120s", task.name, result)
        except Exception as exc:
            logger.warning("[TaskScheduler] task %s execution failed: %s", task.name, exc)

    # ----------------------------------------------------------------- persist

    def _load(self) -> None:
        if not self._file.exists():
            return
        try:
            data = json.loads(self._file.read_text(encoding="utf-8"))
            for row in data.get("tasks", []):
                try:
                    t = ScheduledTask.from_dict(row)
                    self._tasks[t.name] = t
                except Exception:
                    pass
        except Exception:
            pass

    def _save(self) -> None:
        try:
            payload = {"tasks": [t.to_dict() for t in self._tasks.values()]}
            self._file.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _seed_defaults(self) -> None:
        """Pre-seed a daily briefing task if none exists yet."""
        if "daily_briefing" not in self._tasks:
            self._tasks["daily_briefing"] = ScheduledTask(
                name="daily_briefing",
                action="Give me my daily briefing.",
                schedule="daily@09:00",
                enabled=False,  # opt-in: Gabriel enables via /schedule or conversation
                description="Morning situational briefing at 09:00",
            )
            self._save()


_scheduler: Optional[TaskScheduler] = None


def get_task_scheduler() -> TaskScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = TaskScheduler()
    return _scheduler
