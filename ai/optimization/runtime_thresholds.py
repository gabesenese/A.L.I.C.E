"""
Runtime thresholds for A.L.I.C.E (router confidence, policy).
Loaded from data/training/thresholds.json so the offline loop can update them.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = {
    "tool_path_confidence": 0.7,  # Above this -> tool path (plugins, code, etc.)
    "goal_path_confidence": 0.6,
    "ask_threshold": 0.5,
    "router_clarification_threshold": 0.75,
    "conversation_min_confidence": 0.7,
    "unknown_fallback_conf_hard": 0.35,
    "unknown_fallback_conf_soft": 0.45,
    "unknown_fallback_plaus_soft": 0.60,
    "unknown_fallback_plaus_hard": 0.45,
    "route_uncertainty_threshold": 0.55,
    "clarification_intent_confidence_threshold": 0.45,
    "clarification_confidence_min": 0.42,
    "clarification_confidence_max": 0.62,
    "conversation_category_gate_threshold": 0.88,
    "confidence_execute_direct": 0.85,
    "confidence_execute_low": 0.65,
    "confidence_clarify": 0.45,
    "foundation_clarification_confidence": 0.58,
    "foundation_clarification_margin": 0.35,
    "foundation_deep_stage_skip_threshold": 0.95,
}

_threshold_dir = Path("data/training")
_threshold_file = _threshold_dir / "thresholds.json"
_legacy_threshold_file = _threshold_dir / "threshold.json"
_cached: Dict[str, float] = {}


def _load_threshold_file(path: Path) -> Dict[str, float]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return {k: float(v) for k, v in data.items() if k in DEFAULT_THRESHOLDS}
    except Exception as e:
        logger.warning(f"[Thresholds] Failed to load {path}: {e}")
        return {}


def get_thresholds() -> Dict[str, float]:
    """Load thresholds from file: fall back to defaults."""
    global _cached
    if _cached:
        return _cached.copy()

    loaded: Dict[str, float] = {}
    if _threshold_file.exists():
        loaded.update(_load_threshold_file(_threshold_file))
    elif _legacy_threshold_file.exists():
        loaded.update(_load_threshold_file(_legacy_threshold_file))

    _cached = DEFAULT_THRESHOLDS.copy()
    _cached.update(loaded)
    return _cached.copy()


def update_thresholds(updates: Dict[str, float]) -> None:
    """Write updated thresholds to file and refresh cache."""
    global _cached
    current = get_thresholds()
    current.update(
        {
            key: float(value)
            for key, value in (updates or {}).items()
            if key in DEFAULT_THRESHOLDS
        }
    )
    _threshold_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(_threshold_file, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)
        # Keep legacy path in sync for backward compatibility with older tooling.
        with open(_legacy_threshold_file, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)
        _cached = current
        logger.info(f"[Thresholds] Updated: {list(updates.keys())}")
    except Exception as e:
        logger.error(f"[Thresholds] Failed ot save: {e}")


def get_tool_path_confidence() -> float:
    return get_thresholds().get(
        "tool_path_confidence", DEFAULT_THRESHOLDS["tool_path_confidence"]
    )


def get_goal_path_confidence() -> float:
    return get_thresholds().get(
        "goal_path_confidence", DEFAULT_THRESHOLDS["goal_path_confidence"]
    )


def get_ask_threshold() -> float:
    return get_thresholds().get("ask_threshold", DEFAULT_THRESHOLDS["ask_threshold"])


def get_threshold(key: str, default: Any = None) -> float:
    """Read a specific threshold value with typed fallback."""
    thresholds = get_thresholds()
    if key in thresholds:
        return float(thresholds[key])
    if default is None:
        default = DEFAULT_THRESHOLDS.get(key, 0.0)
    return float(default)
