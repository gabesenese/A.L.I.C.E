"""Activity monitor for lightweight proactive suggestions."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ActivitySnapshot:
    last_seen: Dict[str, float]


class ActivityMonitor:
    def __init__(self) -> None:
        self._last_seen: Dict[str, float] = {}

    def observe(self, activity: str) -> None:
        self._last_seen[str(activity or "unknown")] = time.time()

    def snapshot(self) -> ActivitySnapshot:
        return ActivitySnapshot(last_seen=dict(self._last_seen))

    def proactive_suggestions(self, now_ts: float | None = None) -> List[str]:
        now_ts = float(now_ts or time.time())
        out: List[str] = []

        email_ts = self._last_seen.get("email")
        if email_ts and (now_ts - email_ts) > 2 * 3600:
            out.append(
                "You have not checked email for a while. Want a quick inbox summary?"
            )

        study_ts = self._last_seen.get("study")
        if study_ts and (now_ts - study_ts) > 3 * 3600:
            out.append(
                "You have been in study mode for a while. Want a short break reminder?"
            )

        debug_ts = self._last_seen.get("debug")
        if debug_ts and (now_ts - debug_ts) < 30 * 60:
            out.append(
                "I noticed repeated debugging activity. Want me to summarize likely root causes?"
            )

        return out
