from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from memory.world_model import WorldModel, get_world_model


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_time(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _age_seconds(value: Any, now: datetime) -> Optional[float]:
    parsed = _parse_time(value)
    if not parsed:
        return None
    return max(0.0, (now - parsed).total_seconds())


@dataclass(frozen=True)
class HeartbeatDecision:
    key: str
    message: str
    reason: str
    priority: str = "normal"
    metadata: Dict[str, Any] | None = None


@dataclass(frozen=True)
class HeartbeatConfig:
    interval_seconds: float = 60.0
    min_interrupt_gap_seconds: float = 10 * 60
    stale_intention_seconds: float = 2 * 60 * 60
    checkin_gap_seconds: float = 90 * 60
    active_recent_seconds: float = 2 * 60 * 60
    stale_task_seconds: float = 24 * 60 * 60


class Heartbeat:
    """Background companion heartbeat.

    Reads the persistent world model and emits safe CLI interrupts when a
    proactive condition is met. It never executes tools.
    """

    def __init__(
        self,
        *,
        world_model: WorldModel | None = None,
        config: HeartbeatConfig | None = None,
        output: Callable[[str], None] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.world_model = world_model or get_world_model()
        self.config = config or HeartbeatConfig()
        self.output = output or self._stdout_output
        self.clock = clock or _now
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._last_decisions: List[HeartbeatDecision] = []

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return bool(thread and thread.is_alive())

    def start(self) -> None:
        with self._lock:
            if self.is_running:
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._loop,
                name="alice-heartbeat",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        thread = None
        with self._lock:
            self._stop_event.set()
            thread = self._thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2)
        with self._lock:
            self._thread = None

    def run_once(self) -> List[HeartbeatDecision]:
        with self._lock:
            decisions = self.evaluate()
            emitted: List[HeartbeatDecision] = []
            for decision in decisions:
                if not self._can_emit(decision):
                    continue
                self.output(self._format_message(decision.message))
                self.world_model.record_proactive_interrupt(
                    key=decision.key,
                    text=decision.message,
                    source=decision.reason,
                    timestamp=self.clock().isoformat(),
                )
                if decision.reason == "general_checkin_due":
                    self.world_model.set_last_checkin(self.clock().isoformat())
                emitted.append(decision)
                break
            self._last_decisions = emitted
            return emitted

    def evaluate(self) -> List[HeartbeatDecision]:
        snapshot = self.world_model.snapshot()
        user = dict(snapshot.get("user") or {})
        env = dict(snapshot.get("environment") or {})
        now = self.clock()
        decisions: List[HeartbeatDecision] = []

        for item in list(user.get("mentioned_intentions") or []):
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            age = _age_seconds(item.get("created_at"), now)
            if age is not None and age >= self.config.stale_intention_seconds:
                key = f"intention:{text.lower()}"
                decisions.append(
                    HeartbeatDecision(
                        key=key,
                        message=f"Checking in on this: {text}. Still something you want to move forward?",
                        reason="stale_intention",
                        priority="normal",
                        metadata={"age_seconds": age},
                    )
                )

        last_checkin_age = _age_seconds(user.get("last_checkin"), now)
        last_active_age = _age_seconds(user.get("last_active"), now)
        if (
            last_active_age is not None
            and last_active_age <= self.config.active_recent_seconds
            and (
                last_checkin_age is None
                or last_checkin_age >= self.config.checkin_gap_seconds
            )
        ):
            decisions.append(
                HeartbeatDecision(
                    key="checkin:general",
                    message="Quick check-in: how are things going?",
                    reason="general_checkin_due",
                    priority="low",
                    metadata={
                        "last_active_age_seconds": last_active_age,
                        "last_checkin_age_seconds": last_checkin_age,
                    },
                )
            )

        for item in list(env.get("open_tasks") or []):
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            age = _age_seconds(item.get("created_at"), now)
            if age is not None and age >= self.config.stale_task_seconds:
                key = f"task:{text.lower()}"
                decisions.append(
                    HeartbeatDecision(
                        key=key,
                        message=f"This open task has been sitting for a while: {text}.",
                        reason="stale_open_task",
                        priority="normal",
                        metadata={"age_seconds": age},
                    )
                )

        return decisions

    def status(self) -> Dict[str, Any]:
        return {
            "running": self.is_running,
            "interval_seconds": float(self.config.interval_seconds),
            "last_decisions": [
                {
                    "key": d.key,
                    "message": d.message,
                    "reason": d.reason,
                    "priority": d.priority,
                    "metadata": dict(d.metadata or {}),
                }
                for d in self._last_decisions
            ],
        }

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self.run_once()
            self._stop_event.wait(max(1.0, float(self.config.interval_seconds or 60.0)))

    def _can_emit(self, decision: HeartbeatDecision) -> bool:
        snapshot = self.world_model.snapshot()
        alice_state = dict(snapshot.get("alice_state") or {})
        now = self.clock()
        gap = _age_seconds(alice_state.get("last_proactive_interrupt"), now)
        if gap is not None and gap < self.config.min_interrupt_gap_seconds:
            return False
        if self.world_model.has_unresolved_interrupt(decision.key):
            return False
        return True

    @staticmethod
    def _format_message(message: str) -> str:
        return f"\nA.L.I.C.E: {str(message or '').strip()}"

    @staticmethod
    def _stdout_output(text: str) -> None:
        print(text, file=sys.stdout, flush=True)
