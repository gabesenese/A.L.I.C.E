"""Shared fallback policy utilities for runtime response handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


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


@dataclass
class FallbackStep:
    action: str  # "clarify", "retry_with_location", "use_llm", "escalate"
    message: str  # Human-readable recovery suggestion for the user
    requires_user: bool = False  # Whether user input is needed before retrying


class FallbackGraph:
    """Maps (tool_name_or_intent_prefix, error_type) → ordered recovery steps.

    On tool failure the pipeline can walk through steps in order until one succeeds.
    """

    # (intent_prefix, error_type) → [FallbackStep, ...]
    _GRAPH: Dict[tuple, List[FallbackStep]] = {
        ("weather", "no_location"): [
            FallbackStep(
                "clarify_location",
                "What city should I check the weather for?",
                requires_user=True,
            ),
        ],
        ("weather", "unknown_location"): [
            FallbackStep(
                "clarify_location",
                "I couldn't find that location. Could you try a nearby city?",
                requires_user=True,
            ),
        ],
        ("weather", "timeout"): [
            FallbackStep(
                "retry", "Weather service timed out — retrying.", requires_user=False
            ),
            FallbackStep(
                "escalate",
                "Weather service is unreachable right now. Try again in a moment.",
                requires_user=False,
            ),
        ],
        ("weather", "connection_error"): [
            FallbackStep(
                "escalate",
                "Can't reach the weather service. Check your connection.",
                requires_user=False,
            ),
        ],
        ("weather", "fetch_failed"): [
            FallbackStep("retry", "Retrying weather fetch.", requires_user=False),
            FallbackStep(
                "escalate",
                "Weather data isn't available right now.",
                requires_user=False,
            ),
        ],
        ("notes", "not_found"): [
            FallbackStep(
                "clarify_title",
                "Which note did you mean? You can say 'list notes' to see them all.",
                requires_user=True,
            ),
        ],
        ("notes", "ambiguous"): [
            FallbackStep(
                "clarify_title",
                "A few notes match that — which one did you mean?",
                requires_user=True,
            ),
        ],
        ("local", "target_not_found"): [
            FallbackStep(
                "suggest_alternatives",
                "That file wasn't found. Try 'list files' to see what's available.",
                requires_user=True,
            ),
        ],
        ("default", "timeout"): [
            FallbackStep(
                "retry", "Request timed out — retrying once.", requires_user=False
            ),
            FallbackStep(
                "escalate",
                "That's taking too long. Try rephrasing.",
                requires_user=False,
            ),
        ],
        ("default", "tool_failed"): [
            FallbackStep(
                "use_llm",
                "Falling back to language model response.",
                requires_user=False,
            ),
        ],
    }

    def get_steps(self, intent: str, error_type: str) -> List[FallbackStep]:
        """Return ordered recovery steps for the given intent + error combination."""
        intent_prefix = str(intent or "").split(":")[0].lower()
        err = str(error_type or "").lower()

        # Exact match first
        steps = self._GRAPH.get((intent_prefix, err))
        if steps:
            return list(steps)

        # Intent-specific default
        steps = self._GRAPH.get((intent_prefix, "default"))
        if steps:
            return list(steps)

        # Global default
        return list(
            self._GRAPH.get(
                ("default", err), self._GRAPH.get(("default", "tool_failed"), [])
            )
        )

    def first_user_message(self, intent: str, error_type: str) -> Optional[str]:
        """Return the first recovery message for immediate user display, or None."""
        steps = self.get_steps(intent, error_type)
        for step in steps:
            if step.message:
                return step.message
        return None


_fallback_graph: FallbackGraph | None = None


def get_fallback_graph() -> FallbackGraph:
    global _fallback_graph
    if _fallback_graph is None:
        _fallback_graph = FallbackGraph()
    return _fallback_graph


@dataclass
class _FailureRecord:
    count: int = 0
    last_step_index: int = 0  # which FallbackStep we last tried


class RetryMemory:
    """Tracks per-session failure history to escalate recovery strategy on repeat failures.

    On first failure: return step 0.
    On second failure (same error): return step 1 (if available).
    On third+ failure: escalate — tell the user the repeated attempts are failing.
    """

    _MAX_HISTORY = 64

    def __init__(self) -> None:
        self._failures: Dict[tuple, _FailureRecord] = {}

    def _key(self, user_id: str, intent: str, error_type: str) -> tuple:
        prefix = str(intent or "").split(":")[0].lower()
        return (str(user_id or "default"), prefix, str(error_type or "").lower())

    def record_failure(self, user_id: str, intent: str, error_type: str) -> int:
        """Record a failure and return how many times this has now failed."""
        key = self._key(user_id, intent, error_type)
        if len(self._failures) >= self._MAX_HISTORY:
            # Evict oldest
            oldest = next(iter(self._failures))
            del self._failures[oldest]
        rec = self._failures.setdefault(key, _FailureRecord())
        rec.count += 1
        return rec.count

    def get_step_index(self, user_id: str, intent: str, error_type: str) -> int:
        """Return which FallbackStep index to use for this failure (escalates over time)."""
        key = self._key(user_id, intent, error_type)
        rec = self._failures.get(key)
        if rec is None:
            return 0
        # Escalate step index with repeat failures
        return min(rec.count - 1, 2)

    def escalation_message(
        self, user_id: str, intent: str, error_type: str
    ) -> Optional[str]:
        """Return an escalation message if failures have persisted beyond the fallback graph."""
        key = self._key(user_id, intent, error_type)
        rec = self._failures.get(key)
        if rec and rec.count >= 3:
            prefix = str(intent or "").split(":")[0]
            return (
                f"This {prefix} action has failed {rec.count} times in a row. "
                "You may want to try a different approach or check your connection."
            )
        return None

    def clear(self, user_id: str) -> None:
        keys = [k for k in self._failures if k[0] == str(user_id or "default")]
        for k in keys:
            del self._failures[k]


_retry_memory: RetryMemory | None = None


def get_retry_memory() -> RetryMemory:
    global _retry_memory
    if _retry_memory is None:
        _retry_memory = RetryMemory()
    return _retry_memory


def resolve_turn_success_and_route(
    *,
    default_route: str,
    plugin_result: Any,
    llm_attempted: bool,
    llm_generation_success: bool,
    llm_fallback_applied: bool,
) -> Tuple[str, bool, bool]:
    """Resolve the final route/success tuple after execution and fallback.

    Kept in runtime so app/main does not own turn-outcome policy.
    """

    route = str(default_route or "unknown")
    if plugin_result is not None:
        return route, True, False
    if llm_attempted and llm_generation_success:
        return "llm", True, False
    if llm_attempted and not llm_generation_success and llm_fallback_applied:
        return "llm_fallback", True, True
    if llm_attempted:
        return "llm", False, False
    return route, False, False
