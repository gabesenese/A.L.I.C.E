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
    "tool_path_confidence": 0.7, # Above this -> tool path (plugins, code, etc.)
    "goal_path_confidence": 0.6,
    "ask_threshold": 0.5,
}

_threshold_dir = Path("data/training")
_threshold_file = _threshold_dir / "threshold.json"
_cached: Dict[str, float] = {}

def get_thresholds() -> Dict[str, float]:
    """Load thresholds from file: fall back to defaults."""
    global _cached
    if _cached:
        return _cached.copy()
    if _threshold_file.exists():
        try:
            with open(_threshold_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            _cached.update({k: float(v) for k, v in data.items() if k in DEFAULT_THRESHOLDS})
            return _cached.copy()
        except Exception as e:
            logger.warning(f"[Thresholds] Failed to load {_threshold_file}: {e}")
    _cached = DEFAULT_THRESHOLDS.copy()
    return _cached.copy()

def update_thresholds(updates: Dict[str, float]) -> None:
    """Write updated thresholds to file and refresh cache."""
    global _cached
    current = get_thresholds()
    current.update(updates)
    _threshold_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(_threshold_file, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)
        _cached = current
        logger.info(f"[Thresholds] Updated: {list(updates.keys())}")
    except Exception as e:
        logger.error(f"[Thresholds] Failed ot save: {e}")

def get_tool_path_confidence() -> float:
    return get_thresholds().get("tool_path_confidence", DEFAULT_THRESHOLDS["tool_path_confidence"])

def get_goal_path_confidence() -> float:
    return get_thresholds.get("goal_path_confidence", DEFAULT_THRESHOLDS["goal_path_confidence"])

def get_ask_threshold() -> float:
    return get_thresholds().get("ask_threshold", DEFAULT_THRESHOLDS["ask_threshold"])