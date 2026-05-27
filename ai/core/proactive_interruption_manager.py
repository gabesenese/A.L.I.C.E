"""Controls proactive suggestions with cooldowns and user preference respect."""

from __future__ import annotations

import time
from typing import Iterable, List


class ProactiveInterruptionManager:
    def __init__(self, cooldown_seconds: int = 1800, max_suggestions: int = 1) -> None:
        self.cooldown_seconds = max(60, int(cooldown_seconds or 1800))
        self.max_suggestions = max(1, int(max_suggestions or 1))
        self._last_emit_ts = -1.0
        self._suppressed = False

    def set_suppressed(self, suppressed: bool) -> None:
        self._suppressed = bool(suppressed)

    def select(self, suggestions: Iterable[str], now_ts: float | None = None) -> List[str]:
        now_ts = float(now_ts or time.time())
        if self._suppressed:
            return []
        if self._last_emit_ts >= 0.0 and (now_ts - self._last_emit_ts) < self.cooldown_seconds:
            return []

        cleaned = [str(s).strip() for s in suggestions if str(s).strip()]
        if not cleaned:
            return []
        self._last_emit_ts = now_ts
        return cleaned[: self.max_suggestions]
