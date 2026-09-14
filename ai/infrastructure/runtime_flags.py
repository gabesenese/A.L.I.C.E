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
    # Templates that replace an answer the model already produced. See
    # scripted_overrides_enabled() for why this is off rather than on.
    "scripted_overrides",
    # A second generation that rewrites a finished reply. See
    # publish_polish_enabled().
    "publish_polish",
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


def scripted_overrides_enabled() -> bool:
    """Whether a hand-written template may replace an answer the model produced.

    There are two very different things in this codebase that both look like a
    canned string. One *substitutes* for a missing answer — the model was
    unreachable, or returned nothing, and something has to be said. That is a
    fallback, and it stays.

    The other *overrides* an answer that already exists, because it failed a
    shape test: shorter than 70 characters, no comma, no question mark. A real
    reply would be discarded and a template put in its place, which is how a
    direct answer became a menu and a one-line confirmation became an essay.
    That is what this flag governs, and it is off by default: the model's answer
    stands, and grounding checks — not prose heuristics — decide whether it is
    fit to publish.

    Set ALICE_ENABLE_SCRIPTED_OVERRIDES=1 to compare against the old behavior
    with scripts/quality_harness.py.
    """
    return is_enabled("scripted_overrides")


def publish_polish_enabled() -> bool:
    """Whether a finished reply is re-generated to be "natural, clear, and concise".

    The polish pass ran on almost every published response: it took text that had
    already been produced under Alice's persona and asked the model to rewrite it,
    at temperature 0.35, with "concise" in the instruction. Three things follow.
    It costs a whole extra round trip on every turn. A near-deterministic rewrite
    collapses varied phrasings onto the same one, so asking the same thing twice
    gives the same words. And an instruction to compress, applied to every reply
    unconditionally, is a length cap wearing a different hat.

    It is off by default because a second pass over a reply that already carried
    the voice can mostly subtract from it. Set ALICE_ENABLE_PUBLISH_POLISH=1 and
    compare with scripts/quality_harness.py --feel if you want to judge it
    rather than take that on faith.
    """
    return is_enabled("publish_polish")


def background_services_enabled() -> bool:
    """Whether heartbeat, ambient monitoring, and companion daemons may run.

    These threads mutate shared goal, world, and memory state on their own schedule.
    A test process must be able to switch them off, otherwise results depend on
    thread timing rather than on the code under test.
    """
    return is_enabled("background_services")
