"""Turn-state diff utilities for decision and action traceability."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List


def _safe_serialize(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except Exception:
        return str(value)


def generate_turn_diff(
    before: Dict[str, Any] | None,
    after: Dict[str, Any] | None,
    *,
    event: str = "",
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return a compact structural diff between two world-state snapshots."""
    before = dict(before or {})
    after = dict(after or {})

    before_keys = set(before.keys())
    after_keys = set(after.keys())
    added = sorted(list(after_keys - before_keys))
    removed = sorted(list(before_keys - after_keys))

    changed: List[str] = []
    changed_values: Dict[str, Dict[str, Any]] = {}
    for key in sorted(before_keys.intersection(after_keys)):
        if _safe_serialize(before.get(key)) != _safe_serialize(after.get(key)):
            changed.append(key)
            changed_values[key] = {
                "before": before.get(key),
                "after": after.get(key),
            }

    before_open = len(list((before.get("open_questions") or [])))
    after_open = len(list((after.get("open_questions") or [])))
    before_goals = len(list((before.get("active_goals") or [])))
    after_goals = len(list((after.get("active_goals") or [])))

    return {
        "timestamp": time.time(),
        "event": str(event or "turn_update"),
        "added_keys": added,
        "removed_keys": removed,
        "changed_keys": changed,
        "changed_values": changed_values,
        "counts": {
            "before_open_questions": before_open,
            "after_open_questions": after_open,
            "before_active_goals": before_goals,
            "after_active_goals": after_goals,
            "open_question_delta": after_open - before_open,
            "active_goal_delta": after_goals - before_goals,
        },
        "metadata": dict(metadata or {}),
    }
