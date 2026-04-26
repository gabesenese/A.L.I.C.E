"""Runtime feature flags and quarantine defaults.

Subsystems listed in QUARANTINED_SUBSYSTEMS are disabled by default and must
be explicitly enabled via environment variable:
ALICE_ENABLE_<UPPERCASE_NAME>=1
"""

from __future__ import annotations

import os
from typing import Set

QUARANTINED_SUBSYSTEMS: Set[str] = {
    "session_summarizer",
    "capability_constraints",
    "result_quality_scorer",
    "goal_alignment_tracker",
    "tone_trajectory_engine",
    "pattern_based_nudger",
    "system_state_api",
    "weak_spot_detector",
    "multi_goal_arbitrator",
    "routing_decision_logger",
}


def is_enabled(name: str) -> bool:
    """Return True when subsystem is enabled under quarantine policy."""
    key = f"ALICE_ENABLE_{name.upper()}"
    raw = str(os.getenv(key, "")).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return name not in QUARANTINED_SUBSYSTEMS
