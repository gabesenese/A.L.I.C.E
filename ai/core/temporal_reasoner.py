"""Temporal reasoner for schedule-like language and recurring cadence extraction."""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

try:
    dateparser = importlib.import_module("dateparser")
except Exception:
    dateparser = None


@dataclass
class TemporalTask:
    when_iso: str
    frequency: str
    action: str
    confidence: float


class TemporalReasoner:
    def parse_temporal_task(self, text: str, *, now: Optional[datetime] = None) -> Optional[TemporalTask]:
        text = (text or "").strip()
        if not text:
            return None
        now = now or datetime.now()
        lower = text.lower()

        frequency = "once"
        if re.search(r"\b(every day|daily|each day)\b", lower):
            frequency = "daily"
        elif re.search(r"\b(every week|weekly|each week)\b", lower):
            frequency = "weekly"
        elif re.search(r"\b(every month|monthly|each month)\b", lower):
            frequency = "monthly"

        when = None
        if dateparser is not None:
            try:
                when = dateparser.parse(text, settings={"PREFER_DATES_FROM": "future"})
            except Exception:
                when = None

        if when is None:
            if "tomorrow" in lower:
                when = now.replace(hour=9, minute=0, second=0, microsecond=0)
                when = when.fromtimestamp(when.timestamp() + 86400)
            elif re.search(r"\bnext\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", lower):
                # Best-effort fallback: unresolved exact weekday without dateparser.
                when = now.replace(hour=9, minute=0, second=0, microsecond=0)

        if when is None and frequency != "once":
            hour, minute = self._extract_time_of_day(lower)
            when = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if when <= now:
                when = when.fromtimestamp(when.timestamp() + 86400)

        if when is None:
            return None

        action = self._extract_action(lower) or "reminder"
        confidence = 0.86 if dateparser is not None else 0.68
        if frequency != "once":
            confidence = min(0.93, confidence + 0.04)

        return TemporalTask(
            when_iso=when.isoformat(),
            frequency=frequency,
            action=action,
            confidence=confidence,
        )

    def _extract_action(self, lower: str) -> str:
        m = re.search(r"(?:remind me to|remind me about|schedule|set a reminder(?: for)?|every\s+\w+\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s+(.+)", lower)
        if not m:
            return ""
        action = m.group(1).strip(" .!?")
        return action[:120]

    def _extract_time_of_day(self, lower: str) -> tuple[int, int]:
        m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", lower)
        if not m:
            return 9, 0
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        meridian = m.group(3).lower()
        if meridian == "pm" and hour != 12:
            hour += 12
        if meridian == "am" and hour == 12:
            hour = 0
        return max(0, min(23, hour)), max(0, min(59, minute))
