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
import math
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SLOT_MISS = "SLOT_MISS"
FRAME_MISS = "FRAME_MISS"
COREF_MISS = "COREF_MISS"
ROUTE_MISS = "ROUTE_MISS"
PLUGIN_MISS = "PLUGIN_MISS"
CONFIDENCE = "CONFIDENCE"

# NLP-specific failure tags (Improvement 4)
NLP_WRONG_INTENT = "NLP_WRONG_INTENT"  # classified intent != expected intent
NLP_WRONG_ENTITY = "NLP_WRONG_ENTITY"  # entity extraction produced wrong value
BAD_FOLLOWUP = "BAD_FOLLOWUP"  # follow-up resolution did not carry context
BAD_TEMPORAL = "BAD_TEMPORAL"  # temporal expression was misparse or absent

_ALL_TYPES = {
    SLOT_MISS,
    FRAME_MISS,
    COREF_MISS,
    ROUTE_MISS,
    PLUGIN_MISS,
    CONFIDENCE,
    NLP_WRONG_INTENT,
    NLP_WRONG_ENTITY,
    BAD_FOLLOWUP,
    BAD_TEMPORAL,
}

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
    _COREF_TOKENS = frozenset(
        [
            "it",
            "this",
            "that",
            "the note",
            "the file",
            "the one",
            "the first",
            "the second",
            "the last",
            "the previous",
        ]
    )

    def __init__(self, log_path: Optional[str] = None):
        if log_path is None:
            log_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "data",
                "analytics",
                "failure_log.jsonl",
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
        modifiers = (
            parsed_cmd.get("modifiers", {}) if isinstance(parsed_cmd, dict) else {}
        )
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
        if frame_name is None or (
            isinstance(frame, dict) and frame.get("confidence", 1.0) < 0.50
        ):
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
            logger.debug(
                "[FAILURE] %s | %s | %s",
                record.failure_type,
                record.intent,
                record.utterance[:60],
            )
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

    def per_intent_failures(self, n: int = 1000) -> Dict[str, Dict[str, int]]:
        """
        Aggregate failure counts grouped by (intent, failure_type).

        Returns::

            {
              "notes:create": {
                "SLOT_MISS": 12,
                "FRAME_MISS": 3,
                ...
              }, ...
            }
        """
        records = self.load_recent(n)
        result: Dict[str, Dict[str, int]] = {}
        for r in records:
            intent = r.get("intent", "unknown")
            ft = r.get("failure_type", "UNKNOWN")
            bucket = result.setdefault(intent, {})
            bucket[ft] = bucket.get(ft, 0) + 1
        return result

    def generate_nlp_report(self, n: int = 1000) -> Dict[str, Any]:
        """
        Aggregate NLP failures and produce a structured report suitable for
        writing to ``reports/nlp_status.json``.

        Report schema::

            {
              "generated_at": "2025-…",
              "records_analysed": 450,
              "failure_type_summary": {…},
              "top_failing_intents": [
                {"intent": "notes:search", "total_failures": 34,
                 "dominant_type": "SLOT_MISS"}, …   (top 3)
              ],
              "top_failing_patterns": [
                {"failure_type": "SLOT_MISS", "count": 78}, …   (top 3)
              ],
              "suggested_changes": [
                "Add 'query' slot extractor for notes:search (SLOT_MISS ×34)",
                …
              ]
            }
        """
        records = self.load_recent(n)
        type_summary = self.failure_summary()
        per_intent = self.per_intent_failures(n)

        # Top-3 failing intents by total failures
        intent_totals = {
            intent: sum(counts.values()) for intent, counts in per_intent.items()
        }
        top_intents = sorted(intent_totals.items(), key=lambda x: x[1], reverse=True)[
            :3
        ]
        top_failing_intents = []
        for intent, total in top_intents:
            dominant_type = max(per_intent[intent].items(), key=lambda x: x[1])[0]
            top_failing_intents.append(
                {
                    "intent": intent,
                    "total_failures": total,
                    "dominant_type": dominant_type,
                }
            )

        # Top-3 failing patterns by failure_type count
        top_patterns = sorted(type_summary.items(), key=lambda x: x[1], reverse=True)[
            :3
        ]
        top_failing_patterns = [
            {"failure_type": ft, "count": cnt} for ft, cnt in top_patterns if cnt > 0
        ]

        # Suggested changes (heuristic rules)
        suggested: List[str] = []
        for item in top_failing_intents:
            intent = item["intent"]
            dtype = item["dominant_type"]
            total = item["total_failures"]
            if dtype == SLOT_MISS:
                # Extract which slots are most often missing
                missing_counts: Dict[str, int] = {}
                for r in records:
                    if r.get("intent") == intent and r.get("failure_type") == SLOT_MISS:
                        for s in r.get("missing_slots", []):
                            missing_counts[s] = missing_counts.get(s, 0) + 1
                top_slot = (
                    max(missing_counts, key=missing_counts.get)
                    if missing_counts
                    else "unknown"
                )
                suggested.append(
                    f"Add '{top_slot}' slot extractor for {intent} ({dtype} ×{total})"
                )
            elif dtype == FRAME_MISS:
                suggested.append(
                    f"Expand FrameDefinition trigger keywords/patterns for {intent} ({dtype} ×{total})"
                )
            elif dtype == COREF_MISS:
                suggested.append(
                    f"Improve coreference resolution for {intent} context chain ({dtype} ×{total})"
                )
            elif dtype == NLP_WRONG_INTENT:
                suggested.append(
                    f"Add training examples or adjust PHASE 1 patterns for {intent} ({dtype} ×{total})"
                )
            elif dtype == BAD_TEMPORAL:
                suggested.append(
                    f"Improve TemporalParser coverage for {intent} expressions ({dtype} ×{total})"
                )
            else:
                suggested.append(f"Investigate {dtype} failures in {intent} (×{total})")

        return {
            "generated_at": datetime.utcnow().isoformat(),
            "records_analysed": len(records),
            "failure_type_summary": {k: v for k, v in type_summary.items() if v > 0},
            "top_failing_intents": top_failing_intents,
            "top_failing_patterns": top_failing_patterns,
            "suggested_changes": suggested,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_missing_slots(action: str, data: Dict[str, Any]) -> List[str]:
        """Infer which slots are missing based on action + error data."""
        missing: List[str] = []
        clarify_q = (
            data.get("clarification_question", "").lower()
            if isinstance(data, dict)
            else ""
        )
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


# ---------------------------------------------------------------------------
# Self-Debugger  (post-turn root-cause analysis + fix scheduler)
# ---------------------------------------------------------------------------

PHRASING_MISS = "PHRASING_MISS"
MEMORY_MISS = "MEMORY_MISS"


@dataclass
class TurnPostmortem:
    """All observable signals from a completed turn."""

    utterance: str
    intent: str
    plugin: str
    action: str
    confidence: float
    missing_slots: List[str] = field(default_factory=list)
    plugin_success: bool = True
    plugin_error_code: Optional[str] = None
    user_corrected_immediately: bool = False
    response_repeated_request: bool = False
    memory_retrieval_failures: int = 0
    coref_failures: int = 0
    frame_parse_score: float = 1.0
    session_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ThoughtTrace:
    """
    Compact per-turn pipeline trace emitted in dev mode.

    Enabled by the ``thought_trace`` feature flag.  One ``ThoughtTrace`` is
    created per call to ``NLPProcessor.process()`` and logged at DEBUG level
    (and optionally to ``data/analytics/thought_trace.jsonl``).
    """

    turn_index: int
    raw_text: str
    normalized_text: str
    intent: str
    intent_conf: float
    frame_name: Optional[str]
    frame_conf: float
    followup_detected: bool
    frame_merged: bool
    # Which routing/correction path produced the final intent:
    # "learned" | "fingerprint" | "retrieval" | "semantic" | "phase1" | "unknown"
    correction_source: str
    dialogue_state: str
    policy_source: str  # "hand_tuned" | "learned_model" | "blend"
    emotions: List[str]
    urgency: str
    ts: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def compact(self) -> str:
        """One-line summary suitable for a DEBUG log."""
        return (
            f"[{self.turn_index}] intent={self.intent} conf={self.intent_conf:.2f} "
            f"frame={self.frame_name or 'none'} src={self.correction_source} "
            f"state={self.dialogue_state} policy={self.policy_source}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "intent": self.intent,
            "intent_conf": self.intent_conf,
            "frame_name": self.frame_name,
            "frame_conf": self.frame_conf,
            "followup_detected": self.followup_detected,
            "frame_merged": self.frame_merged,
            "correction_source": self.correction_source,
            "dialogue_state": self.dialogue_state,
            "policy_source": self.policy_source,
            "emotions": self.emotions,
            "urgency": self.urgency,
            "ts": self.ts,
        }


class RootCauseClassifier(FailureTaxonomy):
    """
    Extends FailureTaxonomy with two new failure types (PHRASING_MISS,
    MEMORY_MISS) and a classify_postmortem() method.
    """

    def classify_postmortem(self, pm: TurnPostmortem) -> Optional[FailureRecord]:
        failure_type = self._decide(pm)
        if failure_type is None:
            return None
        return FailureRecord(
            failure_type=failure_type,
            utterance=pm.utterance,
            intent=pm.intent,
            plugin=pm.plugin,
            action=pm.action,
            confidence=pm.confidence,
            missing_slots=pm.missing_slots,
            error_code=pm.plugin_error_code,
            extra={
                "memory_retrieval_failures": pm.memory_retrieval_failures,
                "coref_failures": pm.coref_failures,
                "frame_parse_score": pm.frame_parse_score,
                "user_corrected": pm.user_corrected_immediately,
                "response_repeated": pm.response_repeated_request,
            },
        )

    def _decide(self, pm: TurnPostmortem) -> Optional[str]:
        if pm.missing_slots and not pm.plugin_success:
            return SLOT_MISS
        if pm.coref_failures > 0:
            return COREF_MISS
        if pm.frame_parse_score < 0.5:
            return FRAME_MISS
        if not pm.plugin_success:
            if pm.plugin_error_code == "clarification_required":
                return SLOT_MISS
            if pm.confidence < 0.45:
                return ROUTE_MISS
            return PLUGIN_MISS
        if pm.confidence < 0.45 and not pm.plugin_success:
            return CONFIDENCE
        if pm.memory_retrieval_failures > 0:
            return MEMORY_MISS
        if pm.response_repeated_request:
            return PHRASING_MISS
        if pm.user_corrected_immediately:
            return ROUTE_MISS
        return None


@dataclass
class FixAction:
    """A concrete action proposed by the FixScheduler."""

    strategy: str
    failure_type: str
    target: str
    parameter: Any
    confidence: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


FixFn = Callable[[FailureRecord, Dict[str, Any]], bool]


def _fix_threshold_adjust(record: FailureRecord, components: Dict[str, Any]) -> bool:
    classifier = components.get("intent_classifier")
    if not classifier:
        return False
    try:
        thresh_attr = getattr(classifier, "confidence_thresholds", {})
        current = thresh_attr.get(record.intent, 0.5)
        thresh_attr[record.intent] = max(0.2, current - 0.05)
        logger.info(
            "SelfDebugger: lowered threshold for %s → %.2f",
            record.intent,
            thresh_attr[record.intent],
        )
        return True
    except Exception as exc:
        logger.debug("SelfDebugger: threshold_adjust failed: %s", exc)
        return False


def _fix_add_domain_rule(record: FailureRecord, components: Dict[str, Any]) -> bool:
    nlp = components.get("nlp_processor")
    if not nlp:
        return False
    try:
        hints = getattr(nlp, "_debug_hints", [])
        hints.append({"utterance": record.utterance, "expected_intent": record.intent})
        nlp._debug_hints = hints
        logger.info(
            "SelfDebugger: added domain rule hint for %r → %s",
            record.utterance[:50],
            record.intent,
        )
        return True
    except Exception as exc:
        logger.debug("SelfDebugger: add_domain_rule failed: %s", exc)
        return False


def _fix_alter_planner_weight(
    record: FailureRecord, components: Dict[str, Any]
) -> bool:
    planner = components.get("task_planner")
    if not planner:
        return False
    try:
        weights = getattr(planner, "_action_weights", {})
        key = f"{record.plugin}:{record.action}"
        weights[key] = min(2.0, weights.get(key, 1.0) + 0.1)
        planner._action_weights = weights
        logger.info("SelfDebugger: nudged planner weight %s → %.2f", key, weights[key])
        return True
    except Exception as exc:
        logger.debug("SelfDebugger: alter_planner_weight failed: %s", exc)
        return False


def _fix_phrasing_swap(record: FailureRecord, components: Dict[str, Any]) -> bool:
    formulator = components.get("response_formulator")
    if not formulator:
        return False
    try:
        bad = getattr(formulator, "_deprioritised_templates", set())
        intent_key = f"{record.intent}:default"
        bad.add(intent_key)
        formulator._deprioritised_templates = bad
        logger.info("SelfDebugger: deprioritised phrasing template %s", intent_key)
        return True
    except Exception as exc:
        logger.debug("SelfDebugger: phrasing_swap failed: %s", exc)
        return False


def _fix_memory_search_boost(record: FailureRecord, components: Dict[str, Any]) -> bool:
    mem_sys = components.get("memory_system")
    if not mem_sys:
        return False
    try:
        current = getattr(mem_sys, "search_top_k", 5)
        mem_sys.search_top_k = min(current + 2, 20)
        logger.info(
            "SelfDebugger: boosted memory search top-k → %d", mem_sys.search_top_k
        )
        return True
    except Exception as exc:
        logger.debug("SelfDebugger: memory_search_boost failed: %s", exc)
        return False


_FAILURE_STRATEGY_MAP: Dict[str, List[str]] = {
    SLOT_MISS: ["threshold_adjust", "add_domain_rule"],
    ROUTE_MISS: ["threshold_adjust", "add_domain_rule", "alter_planner_weight"],
    FRAME_MISS: ["add_domain_rule", "threshold_adjust"],
    COREF_MISS: ["memory_search_boost", "add_domain_rule"],
    PLUGIN_MISS: ["alter_planner_weight", "threshold_adjust"],
    CONFIDENCE: ["threshold_adjust", "alter_planner_weight"],
    PHRASING_MISS: ["phrasing_swap", "threshold_adjust"],
    MEMORY_MISS: ["memory_search_boost", "add_domain_rule"],
}

_STRATEGY_FNS: Dict[str, FixFn] = {
    "threshold_adjust": _fix_threshold_adjust,
    "add_domain_rule": _fix_add_domain_rule,
    "alter_planner_weight": _fix_alter_planner_weight,
    "phrasing_swap": _fix_phrasing_swap,
    "memory_search_boost": _fix_memory_search_boost,
}


@dataclass
class _StrategyArm:
    successes: int = 0
    failures: int = 0

    def update(self, success: bool) -> None:
        if success:
            self.successes += 1
        else:
            self.failures += 1

    def ucb_score(self, total_pulls: int, c: float = 1.41) -> float:
        pulls = self.successes + self.failures
        if pulls == 0:
            return float("inf")
        mean = self.successes / pulls
        exploration = c * math.sqrt(math.log(total_pulls + 1) / pulls)
        return mean + exploration


class FixScheduler:
    """Multi-armed bandit (UCB1) over fix strategies, per failure type."""

    def __init__(self, log_path: Optional[str] = None) -> None:
        self._arms: Dict[str, _StrategyArm] = {}
        self._total_pulls: int = 0
        self._sched_lock = threading.Lock()
        self._log_path = Path(log_path) if log_path else None

    def schedule(self, record: FailureRecord) -> Optional[FixAction]:
        candidates = _FAILURE_STRATEGY_MAP.get(record.failure_type, [])
        if not candidates:
            return None
        with self._sched_lock:
            self._total_pulls += 1
            best_strategy = max(
                candidates,
                key=lambda s: self._arm(record.failure_type, s).ucb_score(
                    self._total_pulls
                ),
            )
            arm = self._arm(record.failure_type, best_strategy)
            confidence = (arm.successes + 1) / (arm.successes + arm.failures + 2)
        return FixAction(
            strategy=best_strategy,
            failure_type=record.failure_type,
            target=record.intent,
            parameter={"utterance": record.utterance, "plugin": record.plugin},
            confidence=confidence,
        )

    def record_fix_outcome(self, action: FixAction, success: bool) -> None:
        with self._sched_lock:
            self._arm(action.failure_type, action.strategy).update(success)
        if self._log_path:
            self._append_log(action, success)

    def _arm(self, failure_type: str, strategy: str) -> _StrategyArm:
        key = f"{failure_type}:{strategy}"
        if key not in self._arms:
            self._arms[key] = _StrategyArm()
        return self._arms[key]

    def _append_log(self, action: FixAction, success: bool) -> None:
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "timestamp": action.timestamp,
                "strategy": action.strategy,
                "failure_type": action.failure_type,
                "target": action.target,
                "success": success,
            }
            with open(self._log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except Exception as exc:
            logger.debug("FixScheduler: log write failed: %s", exc)


class SelfDebugger:
    """
    Wires RootCauseClassifier → FixScheduler → fix application for each turn.

    Typical call site (post-turn):
        debugger.analyse_turn(postmortem, components_dict)
    """

    def __init__(
        self,
        fix_log_path: Optional[str] = None,
        failure_log_path: Optional[str] = None,
    ) -> None:
        self._classifier = RootCauseClassifier(
            log_path=failure_log_path or "memory/self_debug_failures.jsonl"
        )
        self._scheduler = FixScheduler(
            log_path=fix_log_path or "memory/self_debug_fixes.jsonl"
        )
        self._pending_fixes: List[FixAction] = []

    def analyse_turn(
        self,
        postmortem: TurnPostmortem,
        components: Dict[str, Any],
    ) -> Optional[FixAction]:
        """Classify, schedule, and apply a fix. Returns the FixAction or None."""
        record = self._classifier.classify_postmortem(postmortem)
        if record is None:
            return None
        self._classifier.log(record)
        action = self._scheduler.schedule(record)
        if action is None:
            return None
        fix_fn = _STRATEGY_FNS.get(action.strategy)
        if fix_fn:
            success = fix_fn(record, components)
            self._scheduler.record_fix_outcome(action, success)
            logger.info(
                "SelfDebugger: applied %s for %s → %s",
                action.strategy,
                record.failure_type,
                "ok" if success else "failed",
            )
        return action

    def pending_fixes(self) -> List[FixAction]:
        return list(self._pending_fixes)


_self_debugger_instance: Optional[SelfDebugger] = None
_self_debugger_lock = threading.Lock()


def get_self_debugger() -> SelfDebugger:
    """Return the process-wide singleton SelfDebugger."""
    global _self_debugger_instance
    if _self_debugger_instance is None:
        with _self_debugger_lock:
            if _self_debugger_instance is None:
                _self_debugger_instance = SelfDebugger()
    return _self_debugger_instance
