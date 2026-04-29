"""
A.L.I.C.E. Turn Logger
=======================
Compact JSONL logger that records per-turn features, policy decisions, and
routing outcomes for two consumers:

1. **PolicyTrainer** (policy_trainer.py) — trains a logistic-regression
   model to predict "clarify vs answer" and "cautious vs confident" from
   observable turn features, replacing hand-tuned thresholds in
   InteractionPolicy.

2. **BayesianIntentRouter calibration** — logs top-k intent candidates and
   the final selected intent so per-intent error rates can be computed and
   per-intent thresholds adjusted automatically.

Log format (one JSON object per line)::

    {
      "ts":          "2025-01-01T00:00:00.123456",
      "session_id":  "abc123",
      "turn_index":  5,

      // turn features (inputs to PolicyTrainer)
      "mood":        "frustrated",
      "sentiment":   -0.62,
      "urgency":     "high",
      "intent":      "notes:create",
      "intent_conf": 0.87,
      "frame_name":  "CREATE_NOTE",
      "frame_conf":  0.79,
      "n_slots":     2,

      // policy decision made this turn
      "policy": {
        "clarification_threshold": 0.35,
        "response_length":         "brief",
        "tone":                    "empathetic",
        "skip_clarification":      true,
        "add_empathy_prefix":      true
      },
      "policy_source":  "hand_tuned",   // or "learned_model"

      // router traffic (for per-intent threshold calibration)
      "top_k": [
        {"intent": "notes:create", "score": 0.87},
        {"intent": "notes:append", "score": 0.54}
      ],
      "final_intent": "notes:create",
      "was_corrected": false,   // set to true if user later corrected Alice

      // optional outcome (written retroactively by record_outcome)
      "outcome_reward": null
    }

Usage
-----
    log = TurnLogger()
    entry = TurnLogger.TurnEntry(...)
    log.append(entry)
    log.record_outcome(session_id=..., turn_index=..., reward=1.0)
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default log path
# ---------------------------------------------------------------------------

_DEFAULT_LOG_PATH = Path("data") / "analytics" / "turn_log.jsonl"


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------


@dataclass
class TopKEntry:
    intent: str
    score: float


@dataclass
class PolicySnapshot:
    clarification_threshold: float
    response_length: str
    tone: str
    skip_clarification: bool
    add_empathy_prefix: bool


@dataclass
class TurnEntry:
    """All observable signals captured at the end of one NLP turn."""

    # Identifiers
    session_id: str = ""
    turn_index: int = 0

    # Turn features
    mood: str = "neutral"
    sentiment: float = 0.0  # VADER compound score
    urgency: str = "none"
    intent: str = "conversation:general"
    intent_conf: float = 0.0
    frame_name: Optional[str] = None
    frame_conf: float = 0.0
    n_slots: int = 0

    # Policy
    policy: Optional[PolicySnapshot] = None
    policy_source: str = "hand_tuned"  # "hand_tuned" | "learned_model"

    # Router traffic (for per-intent calibration)
    top_k: List[TopKEntry] = field(default_factory=list)
    final_intent: str = "conversation:general"
    was_corrected: bool = False

    # Outcome (filled retroactively)
    outcome_reward: Optional[float] = None

    # Timestamp
    ts: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Flatten PolicySnapshot
        if d.get("policy"):
            d["policy"] = {k: v for k, v in d["policy"].items()}
        # Flatten TopKEntry list
        d["top_k"] = [{"intent": e["intent"], "score": e["score"]} for e in d["top_k"]]
        return d


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


class TurnLogger:
    """
    Thread-safe JSONL turn logger.

    Features
    --------
    - Append-only writes (one JSON line per turn)
    - ``record_outcome()`` patches the ``outcome_reward`` field in the most
      recent matching entry (in-place rewrite of the last *window* lines)
    - ``load_recent(n)`` reads the last *n* entries for trainer consumption
    - Module-level singleton via ``get_turn_logger()``
    """

    # How many tail lines to scan when back-patching an outcome
    _OUTCOME_SCAN_LINES = 200

    def __init__(self, log_path: Optional[str] = None) -> None:
        if log_path is None:
            log_path = str(_DEFAULT_LOG_PATH)
        self.log_path = Path(log_path).resolve()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def append(self, entry: TurnEntry) -> None:
        """Write one turn entry to the log."""
        try:
            with self._lock:
                with self.log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry.as_dict(), ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("[TurnLogger] Could not write turn log: %s", exc)

    def record_outcome(
        self,
        session_id: str,
        turn_index: int,
        reward: float,
    ) -> bool:
        """
        Back-patch ``outcome_reward`` for the matching (session_id, turn_index)
        entry within the last ``_OUTCOME_SCAN_LINES`` lines.

        Returns True if the entry was found and updated.
        """
        if not self.log_path.exists():
            return False

        try:
            with self._lock:
                with self.log_path.open("r", encoding="utf-8") as fh:
                    all_lines = fh.readlines()

                tail_start = max(0, len(all_lines) - self._OUTCOME_SCAN_LINES)
                updated = False

                for i in range(len(all_lines) - 1, tail_start - 1, -1):
                    try:
                        obj = json.loads(all_lines[i])
                    except json.JSONDecodeError:
                        continue
                    if (
                        obj.get("session_id") == session_id
                        and obj.get("turn_index") == turn_index
                        and obj.get("outcome_reward") is None
                    ):
                        obj["outcome_reward"] = round(reward, 4)
                        all_lines[i] = json.dumps(obj, ensure_ascii=False) + "\n"
                        updated = True
                        break

                if updated:
                    with self.log_path.open("w", encoding="utf-8") as fh:
                        fh.writelines(all_lines)

                return updated
        except Exception as exc:
            logger.warning("[TurnLogger] record_outcome failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load_recent(self, n: int = 500) -> List[Dict[str, Any]]:
        """Return the last *n* parsed turn entries."""
        if not self.log_path.exists():
            return []
        try:
            with self._lock:
                with self.log_path.open("r", encoding="utf-8") as fh:
                    lines = fh.readlines()
            records: List[Dict[str, Any]] = []
            for line in lines[-n:]:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
            return records
        except Exception as exc:
            logger.warning("[TurnLogger] load_recent failed: %s", exc)
            return []

    def load_labelled(self, n: int = 2000) -> List[Dict[str, Any]]:
        """Return only entries that have a non-null ``outcome_reward``."""
        return [r for r in self.load_recent(n) if r.get("outcome_reward") is not None]

    def per_intent_stats(self, n: int = 2000) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate per-intent routing stats from the last *n* entries:

            {
              "notes:create": {
                "total":     120,
                "corrected":  8,
                "error_rate": 0.067,
                "avg_conf":   0.83
              }, ...
            }
        """
        records = self.load_recent(n)
        stats: Dict[str, Dict[str, Any]] = {}
        for r in records:
            intent = r.get("final_intent", "unknown")
            s = stats.setdefault(
                intent,
                {"total": 0, "corrected": 0, "conf_sum": 0.0},
            )
            s["total"] += 1
            if r.get("was_corrected"):
                s["corrected"] += 1
            s["conf_sum"] += float(r.get("intent_conf", 0.0))

        result: Dict[str, Dict[str, Any]] = {}
        for intent, s in stats.items():
            total = s["total"]
            result[intent] = {
                "total": total,
                "corrected": s["corrected"],
                "error_rate": round(s["corrected"] / total, 4) if total > 0 else 0.0,
                "avg_conf": round(s["conf_sum"] / total, 4) if total > 0 else 0.0,
            }
        return result


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: Optional[TurnLogger] = None
_instance_lock = threading.Lock()


def get_turn_logger(log_path: Optional[str] = None) -> TurnLogger:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = TurnLogger(log_path)
    return _instance
