"""Reversibility scoring: how easily can an action be undone?

Score 0.0 = permanent/irreversible (e.g. force-push, disk wipe)
Score 1.0 = trivially reversible (e.g. read-only query, list files)

Used by the safety/policy layer to escalate confirmation requirements when
reversibility is low AND the risk is high.
"""

from __future__ import annotations

import re
from typing import Any, Dict


# Maps action keywords → reversibility score
_REVERSIBILITY_TABLE: Dict[str, float] = {
    # Fully reversible
    "list": 1.0,
    "read": 1.0,
    "get": 1.0,
    "show": 1.0,
    "view": 1.0,
    "search": 1.0,
    "query": 1.0,
    "inspect": 1.0,
    "preview": 1.0,
    # Mostly reversible (version-controlled, trash bin, etc.)
    "create": 0.85,
    "add": 0.85,
    "append": 0.85,
    "save": 0.80,
    "write": 0.75,
    "edit": 0.75,
    "update": 0.75,
    "rename": 0.70,
    "move": 0.70,
    "copy": 0.90,
    # Partially reversible
    "install": 0.60,
    "enable": 0.65,
    "disable": 0.65,
    "start": 0.70,
    "stop": 0.70,
    "restart": 0.65,
    "push": 0.55,
    "merge": 0.50,
    "deploy": 0.45,
    # Barely reversible
    "delete": 0.30,
    "remove": 0.30,
    "uninstall": 0.25,
    "drop": 0.15,
    "truncate": 0.10,
    # Irreversible
    "wipe": 0.05,
    "purge": 0.05,
    "destroy": 0.05,
    "format": 0.05,
    "force_push": 0.0,
    "force-push": 0.0,
    "rm -rf": 0.0,
    "overwrite": 0.20,
    "send": 0.10,  # sent messages can't be unsent
    "publish": 0.10,
    "broadcast": 0.05,
}

_ACTION_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_REVERSIBILITY_TABLE, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


class ReversibilityScorer:
    """Score an action's reversibility based on intent string and user input."""

    def score(self, *, intent: str, user_input: str = "", params: Dict[str, Any] | None = None) -> float:
        """Return reversibility in [0, 1].  Lower = harder to undo."""
        text = f"{intent} {user_input}".lower()

        matches = _ACTION_RE.findall(text)
        if not matches:
            return 0.70  # unknown → conservatively partially reversible

        # Use the minimum score found (worst-case reversibility wins)
        scores = [_REVERSIBILITY_TABLE.get(m.lower().replace(" ", "_"), 0.70) for m in matches]
        return round(min(scores), 3)

    def reversibility_label(self, score: float) -> str:
        if score >= 0.80:
            return "reversible"
        if score >= 0.50:
            return "partially_reversible"
        if score >= 0.20:
            return "hard_to_reverse"
        return "irreversible"

    def requires_rollback_plan(self, score: float) -> bool:
        return score < 0.50

    def confidence_floor(self, score: float) -> float:
        """Minimum routing confidence required to execute without extra confirmation."""
        if score >= 0.80:
            return 0.55
        if score >= 0.50:
            return 0.70
        if score >= 0.20:
            return 0.85
        return 0.95  # irreversible → needs very high confidence


_scorer: ReversibilityScorer | None = None


def get_reversibility_scorer() -> ReversibilityScorer:
    global _scorer
    if _scorer is None:
        _scorer = ReversibilityScorer()
    return _scorer
