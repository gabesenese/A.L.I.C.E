"""Detects recurring interaction patterns across the running session."""

from __future__ import annotations

from collections import Counter
from typing import Dict


class CrossSessionPatternDetector:
    def __init__(self) -> None:
        self._intent_counter: Counter[str] = Counter()

    def observe(self, intent: str) -> None:
        name = str(intent or "unknown")
        self._intent_counter[name] += 1

    def summary(self, top_n: int = 5) -> Dict[str, int]:
        top = self._intent_counter.most_common(max(1, int(top_n or 5)))
        return {k: v for k, v in top}
