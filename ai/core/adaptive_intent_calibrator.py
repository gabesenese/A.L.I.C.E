"""Adaptive confidence calibration from explicit user corrections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class IntentStats:
    correct: int = 0
    wrong: int = 0


class AdaptiveIntentCalibrator:
    """Dynamically adjusts confidence using per-intent correction history."""

    def __init__(self) -> None:
        self._stats: Dict[str, IntentStats] = {}

    def record_feedback(self, intent: str, *, was_correct: bool) -> None:
        key = str(intent or "unknown").strip().lower()
        if key not in self._stats:
            self._stats[key] = IntentStats()
        if was_correct:
            self._stats[key].correct += 1
        else:
            self._stats[key].wrong += 1

    def calibrate(self, intent: str, confidence: float) -> float:
        """Reliability-weighted confidence with shrinkage for low sample intents."""
        key = str(intent or "unknown").strip().lower()
        stats = self._stats.get(key)
        base = max(0.0, min(1.0, float(confidence or 0.0)))
        if stats is None:
            return base

        total = stats.correct + stats.wrong
        if total <= 0:
            return base

        # Laplace-smoothed reliability estimate in [0, 1].
        reliability = (stats.correct + 1.0) / (total + 2.0)
        # More history -> stronger effect.
        strength = min(0.45, total * 0.05)
        adjusted = base * (1.0 - strength) + (base * reliability) * strength
        return max(0.0, min(1.0, adjusted))

    def snapshot(self) -> Dict[str, Dict[str, int | float]]:
        out: Dict[str, Dict[str, int | float]] = {}
        for intent, s in self._stats.items():
            total = s.correct + s.wrong
            rel = (s.correct + 1.0) / (total + 2.0) if total > 0 else 0.5
            out[intent] = {
                "correct": int(s.correct),
                "wrong": int(s.wrong),
                "total": int(total),
                "reliability": float(rel),
            }
        return out
