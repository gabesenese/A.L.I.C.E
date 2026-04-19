"""Shared fallback policy utilities for runtime response handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple


def _normalize(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


@dataclass(frozen=True)
class FallbackDecision:
    action: str
    text: str
    reason: str


class RuntimeFallbackPolicy:
    """Caches deterministic fallbacks per turn and applies refine->fallback order."""

    def __init__(self) -> None:
        self._deterministic_cache: Dict[Tuple[str, str], str] = {}
        self._turn_key: str = ""

    def start_turn(self, turn_key: str = "") -> None:
        self._turn_key = str(turn_key or "")
        self._deterministic_cache.clear()

    def deterministic_once(
        self,
        *,
        user_input: str,
        intent: str,
        builder: Callable[[], Optional[str]],
    ) -> Optional[str]:
        key = (_normalize(user_input), _normalize(intent))
        if key not in self._deterministic_cache:
            generated = str(builder() or "").strip()
            self._deterministic_cache[key] = generated

        value = str(self._deterministic_cache.get(key) or "").strip()
        return value or None

    def refine_then_deterministic(
        self,
        *,
        response: str,
        accepted: bool,
        refine_fn: Callable[[str], Optional[str]],
        deterministic_fn: Callable[[], Optional[str]],
    ) -> FallbackDecision:
        base = str(response or "").strip()

        if accepted and base:
            return FallbackDecision(action="publish", text=base, reason="llm_accepted")

        if base:
            refined = str(refine_fn(base) or "").strip()
            if refined:
                return FallbackDecision(
                    action="refine",
                    text=refined,
                    reason="llm_rejected_refined",
                )

        deterministic = str(deterministic_fn() or "").strip()
        if deterministic:
            return FallbackDecision(
                action="deterministic_fallback",
                text=deterministic,
                reason="llm_rejected_deterministic_fallback",
            )

        if base:
            return FallbackDecision(
                action="defer",
                text="",
                reason="llm_rejected_no_fallback_available",
            )

        return FallbackDecision(
            action="empty",
            text="",
            reason="llm_rejected_empty_no_fallback_available",
        )
