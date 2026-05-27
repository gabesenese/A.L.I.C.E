from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_BUDGET_CHARS = 3_500

# Multiplier applied to a memory's score based on its type.
# Higher-quality memory types get more of the budget.
_TYPE_PRIORITY: Dict[str, float] = {
    "procedural": 1.00,
    "semantic": 0.85,
    "document": 0.70,
    "episodic": 0.60,
}


class RetrievalBudget:
    """
    Budget-aware memory retrieval.

    Given a fixed character budget, selects and orders the best-fitting
    memories without exceeding the budget. Type priority weights are blended
    with per-entry scores (weighted_score > importance > similarity).

    Usage:
        budget = RetrievalBudget(total_chars=3500)
        selected = budget.select(candidates)
        context_str = budget.format_context(selected)
    """

    def __init__(
        self,
        total_chars: int = _DEFAULT_BUDGET_CHARS,
        type_priority: Optional[Dict[str, float]] = None,
    ) -> None:
        self.total_chars = total_chars
        self._type_priority = type_priority or _TYPE_PRIORITY

    def select(
        self,
        candidates: List[Dict[str, Any]],
        *,
        min_priority: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Select memories that fit within the character budget.

        candidates: memory dicts with at minimum 'content' and 'memory_type'
                    (or 'type'); optionally 'weighted_score', 'importance',
                    'similarity'.
        min_priority: discard candidates whose blended priority is below this.

        Returns a list ordered by priority, total content length ≤ total_chars.
        Always includes at least one memory (truncated if necessary).
        """

        def _priority(m: Dict[str, Any]) -> float:
            mtype = str(m.get("memory_type") or m.get("type") or "episodic")
            type_w = self._type_priority.get(mtype, 0.5)
            score = float(m.get("weighted_score") or m.get("importance") or m.get("similarity") or 0.5)
            return type_w * score

        scored = [(m, _priority(m)) for m in candidates if _priority(m) >= min_priority]
        scored.sort(key=lambda x: x[1], reverse=True)

        selected: List[Dict[str, Any]] = []
        remaining = self.total_chars

        for m, _ in scored:
            content = str(m.get("content") or "")
            cost = len(content)

            if cost > remaining:
                if not selected:
                    # Always surface at least one memory — truncate if needed
                    m = dict(m)
                    m["content"] = content[:remaining]
                    selected.append(m)
                break

            selected.append(m)
            remaining -= cost

        if selected:
            used = self.total_chars - remaining
            logger.debug(
                "[RetrievalBudget] %d/%d memories selected, %d/%d chars used",
                len(selected),
                len(candidates),
                used,
                self.total_chars,
            )
        return selected

    def format_context(self, selected: List[Dict[str, Any]]) -> str:
        """Format selected memories into a compact context block for the LLM."""
        if not selected:
            return ""
        parts: List[str] = []
        for m in selected:
            ts = str(m.get("timestamp") or "")[:10]
            mtype = str(m.get("memory_type") or m.get("type") or "memory")
            content = str(m.get("content") or "")
            ws = m.get("weighted_score")
            score_str = f", score={ws:.2f}" if ws is not None else ""
            parts.append(f"[{mtype}, {ts}{score_str}] {content}")
        return "\n".join(parts)


_budget: Optional[RetrievalBudget] = None


def get_retrieval_budget(total_chars: int = _DEFAULT_BUDGET_CHARS) -> RetrievalBudget:
    global _budget
    if _budget is None:
        _budget = RetrievalBudget(total_chars=total_chars)
    return _budget
