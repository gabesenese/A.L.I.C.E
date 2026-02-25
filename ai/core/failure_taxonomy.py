"""
A.L.I.C.E. Failure Taxonomy + Auto-Retraining Hook
=====================================================
Classifies why a turn failed and streams structured failure records to a
JSONL log that the continuous-learning loop can consume.

Failure types
-------------
SLOT_MISS    – intent was correct, but a required slot was absent
FRAME_MISS   – frame parser didn't match or low-confidence frame
COREF_MISS   – pronoun/reference couldn't be resolved
ROUTE_MISS   – intent routing landed on the wrong plugin
PLUGIN_MISS  – plugin raised an error or returned success=False unexpectedly
CONFIDENCE   – overall confidence too low to act reliably

Usage
-----
>>> from ai.core.failure_taxonomy import FailureTaxonomy
>>> tax = FailureTaxonomy()
>>> record = tax.classify("search my notes", intent="notes:search_notes",
...     plugin_result={"success": False, "action": "search_notes",
...                    "data": {"error": "clarification_required"}},
...     nlp_result={"intent_confidence": 0.52, "parsed_command": {}})
>>> tax.log(record)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SLOT_MISS     = "SLOT_MISS"
FRAME_MISS    = "FRAME_MISS"
COREF_MISS    = "COREF_MISS"
ROUTE_MISS    = "ROUTE_MISS"
PLUGIN_MISS   = "PLUGIN_MISS"
CONFIDENCE    = "CONFIDENCE"

_ALL_TYPES = {SLOT_MISS, FRAME_MISS, COREF_MISS, ROUTE_MISS, PLUGIN_MISS, CONFIDENCE}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FailureRecord:
    failure_type: str
    utterance: str
    intent: str
    plugin: str
    action: str
    confidence: float
    missing_slots: List[str] = field(default_factory=list)
    coref_pronoun: Optional[str] = None
    expected_intent: Optional[str] = None
    error_code: Optional[str] = None
    frame_name: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def as_dict(self) -> Dict[str, Any]:
        return {
            "failure_type": self.failure_type,
            "utterance": self.utterance,
            "intent": self.intent,
            "plugin": self.plugin,
            "action": self.action,
            "confidence": round(self.confidence, 4),
            "missing_slots": self.missing_slots,
            "coref_pronoun": self.coref_pronoun,
            "expected_intent": self.expected_intent,
            "error_code": self.error_code,
            "frame_name": self.frame_name,
            "extra": self.extra,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class FailureTaxonomy:
    """
    Classify and log NLP/plugin failures for targeted retraining.
    """

    # Pronouns / referential phrases that indicate a coref attempt
    _COREF_TOKENS = frozenset([
        "it", "this", "that", "the note", "the file", "the one",
        "the first", "the second", "the last", "the previous",
    ])

    def __init__(self, log_path: Optional[str] = None):
        if log_path is None:
            log_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "data", "analytics", "failure_log.jsonl"
            )
        self.log_path = Path(log_path).resolve()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        utterance: str,
        intent: str,
        plugin_result: Optional[Dict[str, Any]],
        nlp_result: Optional[Dict[str, Any]] = None,
    ) -> Optional[FailureRecord]:
        """
        Classify why this turn failed.  Returns None if the turn succeeded.
        """
        plugin_result = plugin_result or {}
        nlp_result = nlp_result or {}

        success = plugin_result.get("success", True)
        if success:
            return None  # Not a failure

        plugin = intent.split(":")[0] if ":" in intent else intent
        action = plugin_result.get("action", "")
        confidence = float(nlp_result.get("intent_confidence", 0.5) or 0.5)
        data = plugin_result.get("data", {}) or {}
        error_code = data.get("error") or data.get("message_code")

        parsed_cmd = nlp_result.get("parsed_command", {}) or {}
        modifiers = parsed_cmd.get("modifiers", {}) if isinstance(parsed_cmd, dict) else {}
        frame = modifiers.get("frame", {}) if isinstance(modifiers, dict) else {}
        frame_name = frame.get("name") if isinstance(frame, dict) else None

        utt_lower = utterance.lower()

        # ── 1. Clarification required → SLOT_MISS ─────────────────────
        if error_code in ("clarification_required", "no_command"):
            missing = self._find_missing_slots(action, data)
            return FailureRecord(
                failure_type=SLOT_MISS,
                utterance=utterance,
                intent=intent,
                plugin=plugin,
                action=action,
                confidence=confidence,
                missing_slots=missing,
                error_code=error_code,
                frame_name=frame_name,
            )

        # ── 2. Pronoun in utterance but resolution failed ──────────────
        if any(tok in utt_lower.split() for tok in self._COREF_TOKENS):
            coref_token = next(
                (tok for tok in self._COREF_TOKENS if tok in utt_lower.split()), None
            )
            # Only classify as COREF_MISS if result doesn't have a valid note_ref
            resolved_ok = bool(
                data.get("note_title") or data.get("query") or data.get("results")
            )
            if not resolved_ok:
                return FailureRecord(
                    failure_type=COREF_MISS,
                    utterance=utterance,
                    intent=intent,
                    plugin=plugin,
                    action=action,
                    confidence=confidence,
                    coref_pronoun=coref_token,
                    error_code=error_code,
                    frame_name=frame_name,
                )

        # ── 3. Low confidence overall → CONFIDENCE ────────────────────
        if confidence < 0.45:
            return FailureRecord(
                failure_type=CONFIDENCE,
                utterance=utterance,
                intent=intent,
                plugin=plugin,
                action=action,
                confidence=confidence,
                error_code=error_code,
                frame_name=frame_name,
            )

        # ── 4. Frame parser had no match or low confidence → FRAME_MISS
        if frame_name is None or (isinstance(frame, dict) and frame.get("confidence", 1.0) < 0.50):
            return FailureRecord(
                failure_type=FRAME_MISS,
                utterance=utterance,
                intent=intent,
                plugin=plugin,
                action=action,
                confidence=confidence,
                error_code=error_code,
                frame_name=frame_name,
            )

        # ── 5. Plugin returned an error code → PLUGIN_MISS ────────────
        if error_code:
            return FailureRecord(
                failure_type=PLUGIN_MISS,
                utterance=utterance,
                intent=intent,
                plugin=plugin,
                action=action,
                confidence=confidence,
                error_code=error_code,
                frame_name=frame_name,
            )

        # ── 6. Generic plugin miss ─────────────────────────────────────
        return FailureRecord(
            failure_type=PLUGIN_MISS,
            utterance=utterance,
            intent=intent,
            plugin=plugin,
            action=action,
            confidence=confidence,
            error_code=error_code,
            frame_name=frame_name,
        )

    def log(self, record: FailureRecord) -> None:
        """Append the failure record to the JSONL log."""
        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")
            logger.debug("[FAILURE] %s | %s | %s", record.failure_type, record.intent, record.utterance[:60])
        except Exception as exc:
            logger.warning("[FAILURE] Could not write failure log: %s", exc)

    def classify_and_log(
        self,
        utterance: str,
        intent: str,
        plugin_result: Optional[Dict[str, Any]],
        nlp_result: Optional[Dict[str, Any]] = None,
    ) -> Optional[FailureRecord]:
        """Classify and immediately log if a failure was detected."""
        record = self.classify(utterance, intent, plugin_result, nlp_result)
        if record:
            self.log(record)
        return record

    def load_recent(self, n: int = 200) -> List[Dict[str, Any]]:
        """Read the last *n* failure records for analysis."""
        if not self.log_path.exists():
            return []
        lines: List[str] = []
        try:
            with self.log_path.open("r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception:
            return []
        records: List[Dict[str, Any]] = []
        for line in lines[-n:]:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return records

    def failure_summary(self) -> Dict[str, int]:
        """Count failures by type from the log."""
        records = self.load_recent(500)
        counts: Dict[str, int] = {t: 0 for t in _ALL_TYPES}
        for r in records:
            ft = r.get("failure_type", "UNKNOWN")
            counts[ft] = counts.get(ft, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_missing_slots(action: str, data: Dict[str, Any]) -> List[str]:
        """Infer which slots are missing based on action + error data."""
        missing: List[str] = []
        clarify_q = data.get("clarification_question", "").lower() if isinstance(data, dict) else ""
        msg_code = data.get("message_code", "") if isinstance(data, dict) else ""

        if "search" in msg_code or "query" in clarify_q:
            missing.append("query")
        if "note_ref" in msg_code or "title" in clarify_q or "which note" in clarify_q:
            missing.append("note_ref")
        if "content" in msg_code or "content" in clarify_q:
            missing.append("content")
        if not missing:
            # Generic: required slots for known actions
            _action_required: Dict[str, List[str]] = {
                "search_notes": ["query"],
                "search_notes_content": ["query"],
                "get_note_content": ["note_ref"],
                "delete_note": ["note_ref"],
                "archive_note": ["note_ref"],
                "update_note": ["note_ref"],
            }
            missing = _action_required.get(action, [])
        return missing


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_instance: Optional[FailureTaxonomy] = None


def get_failure_taxonomy() -> FailureTaxonomy:
    global _instance
    if _instance is None:
        _instance = FailureTaxonomy()
    return _instance
