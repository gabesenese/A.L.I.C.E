"""
tests/integration/test_nlp_error_logger.py
────────────────────────────────────────────
Tests for NLPErrorLogger — the thread-safe JSONL logger that records NLP
correction events so LearningEngine can ingest them as training signals.

Covers:
  • log_intent_override()
  • log_followup_resolved()
  • log_clarification_skip()
  • thread-safety (concurrent writes produce distinct, valid entries)
  • singleton helper get_nlp_error_logger()
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from ai.learning.learning_engine import NLPErrorLogger, get_nlp_error_logger


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def log_file(tmp_path: Path) -> Path:
    return tmp_path / "nlp_errors.jsonl"


@pytest.fixture()
def logger(log_file: Path) -> NLPErrorLogger:
    return NLPErrorLogger(log_file=log_file)


def _read_entries(log_file: Path) -> list[dict]:
    if not log_file.exists():
        return []
    return [
        json.loads(line)
        for line in log_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ── log_intent_override ───────────────────────────────────────────────────────


class TestLogIntentOverride:
    def test_writes_one_entry(self, logger: NLPErrorLogger, log_file: Path):
        logger.log_intent_override(
            user_input="what time is it",
            original_intent="notes:general",
            corrected_intent="time:current",
            original_confidence=0.55,
            corrected_confidence=0.90,
            reason="active_learning",
        )
        entries = _read_entries(log_file)
        assert len(entries) == 1

    def test_entry_schema(self, logger: NLPErrorLogger, log_file: Path):
        logger.log_intent_override(
            user_input="what time is it",
            original_intent="notes:general",
            corrected_intent="time:current",
            original_confidence=0.55,
            corrected_confidence=0.90,
            reason="active_learning",
        )
        entry = _read_entries(log_file)[0]
        assert entry["type"] == "intent_override"
        assert entry["success"] is False
        assert entry["actual_intent"] == "notes:general"
        assert entry["expected_intent"] == "time:current"
        assert entry["domain"] == "notes"
        assert entry["error_type"] == "intent_override"
        assert "timestamp" in entry

    def test_confidence_is_rounded(self, logger: NLPErrorLogger, log_file: Path):
        logger.log_intent_override(
            user_input="x",
            original_intent="a:b",
            corrected_intent="c:d",
            original_confidence=0.123456789,
            corrected_confidence=0.987654321,
            reason="test",
        )
        entry = _read_entries(log_file)[0]
        assert len(str(entry["original_confidence"]).replace("0.", "")) <= 4
        assert len(str(entry["corrected_confidence"]).replace("0.", "")) <= 4

    def test_domain_extracted_from_colon_split(
        self, logger: NLPErrorLogger, log_file: Path
    ):
        logger.log_intent_override(
            user_input="x",
            original_intent="weather:current",
            corrected_intent="y",
            original_confidence=0.5,
            corrected_confidence=0.8,
            reason="r",
        )
        assert _read_entries(log_file)[0]["domain"] == "weather"

    def test_session_id_included_when_provided(
        self, logger: NLPErrorLogger, log_file: Path
    ):
        logger.log_intent_override(
            user_input="x",
            original_intent="a:b",
            corrected_intent="c:d",
            original_confidence=0.5,
            corrected_confidence=0.8,
            reason="r",
            session_id="sess-42",
        )
        assert _read_entries(log_file)[0]["session_id"] == "sess-42"


# ── log_followup_resolved ──────────────────────────────────────────────────────


class TestLogFollowupResolved:
    def test_writes_one_entry(self, logger: NLPErrorLogger, log_file: Path):
        logger.log_followup_resolved(
            user_input="what about tomorrow",
            nlp_intent="conversation:general",
            resolved_intent="weather:forecast",
            nlp_confidence=0.45,
            domain="weather",
            reason="domain_signal",
        )
        entries = _read_entries(log_file)
        assert len(entries) == 1

    def test_entry_schema(self, logger: NLPErrorLogger, log_file: Path):
        logger.log_followup_resolved(
            user_input="what about tomorrow",
            nlp_intent="conversation:general",
            resolved_intent="weather:forecast",
            nlp_confidence=0.45,
            domain="weather",
            reason="domain_signal",
        )
        entry = _read_entries(log_file)[0]
        assert entry["type"] == "followup_resolved"
        assert entry["success"] is False
        assert entry["actual_intent"] == "conversation:general"
        assert entry["expected_intent"] == "weather:forecast"
        assert entry["domain"] == "weather"
        assert entry["error_type"] == "followup_missed"
        assert "timestamp" in entry

    def test_nlp_confidence_is_rounded(self, logger: NLPErrorLogger, log_file: Path):
        logger.log_followup_resolved(
            user_input="x",
            nlp_intent="a",
            resolved_intent="b",
            nlp_confidence=0.123456,
            domain="d",
            reason="r",
        )
        entry = _read_entries(log_file)[0]
        # 4 decimal places max
        assert entry["nlp_confidence"] == round(0.123456, 4)


# ── log_clarification_skip ────────────────────────────────────────────────────


class TestLogClarificationSkip:
    def test_writes_one_entry(self, logger: NLPErrorLogger, log_file: Path):
        logger.log_clarification_skip(
            user_input="help",
            intent="conversation:general",
            confidence=0.72,
            mood="frustrated",
            reason="interaction_policy",
        )
        entries = _read_entries(log_file)
        assert len(entries) == 1

    def test_entry_is_marked_success(self, logger: NLPErrorLogger, log_file: Path):
        logger.log_clarification_skip(
            user_input="help",
            intent="conversation:general",
            confidence=0.72,
            mood="frustrated",
            reason="interaction_policy",
        )
        entry = _read_entries(log_file)[0]
        assert entry["success"] is True

    def test_entry_schema(self, logger: NLPErrorLogger, log_file: Path):
        logger.log_clarification_skip(
            user_input="help",
            intent="notes:search",
            confidence=0.60,
            mood="urgent",
            reason="interaction_policy",
            session_id="s-1",
        )
        entry = _read_entries(log_file)[0]
        assert entry["type"] == "clarification_skip"
        assert entry["domain"] == "notes"
        assert entry["error_type"] == "clarification_skip"
        assert entry["session_id"] == "s-1"


# ── multiple entries ──────────────────────────────────────────────────────────


class TestMultipleEntries:
    def test_each_log_call_appends_a_new_line(
        self, logger: NLPErrorLogger, log_file: Path
    ):
        logger.log_intent_override(
            user_input="a",
            original_intent="x:y",
            corrected_intent="z:w",
            original_confidence=0.5,
            corrected_confidence=0.9,
            reason="r",
        )
        logger.log_followup_resolved(
            user_input="b",
            nlp_intent="p",
            resolved_intent="q",
            nlp_confidence=0.4,
            domain="d",
            reason="r",
        )
        logger.log_clarification_skip(
            user_input="c",
            intent="m:n",
            confidence=0.7,
            mood="neutral",
            reason="r",
        )
        entries = _read_entries(log_file)
        assert len(entries) == 3
        assert entries[0]["type"] == "intent_override"
        assert entries[1]["type"] == "followup_resolved"
        assert entries[2]["type"] == "clarification_skip"


# ── thread safety ─────────────────────────────────────────────────────────────


class TestThreadSafety:
    def test_concurrent_writes_produce_distinct_valid_entries(
        self, logger: NLPErrorLogger, log_file: Path
    ):
        N = 20
        barrier = threading.Barrier(N)

        def _write(i: int) -> None:
            barrier.wait()  # synchronise all threads to start simultaneously
            logger.log_intent_override(
                user_input=f"query {i}",
                original_intent="a:b",
                corrected_intent="c:d",
                original_confidence=0.5,
                corrected_confidence=0.9,
                reason="thread_test",
            )

        threads = [threading.Thread(target=_write, args=(i,)) for i in range(N)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        entries = _read_entries(log_file)
        assert len(entries) == N
        user_inputs = {e["user_input"] for e in entries}
        assert len(user_inputs) == N  # each write preserved its own data


# ── directory auto-creation ───────────────────────────────────────────────────


class TestDirectoryCreation:
    def test_creates_parent_directory_if_missing(self, tmp_path: Path):
        nested = tmp_path / "deep" / "nested" / "nlp_errors.jsonl"
        logger = NLPErrorLogger(log_file=nested)
        logger.log_clarification_skip(
            user_input="x",
            intent="a",
            confidence=0.5,
            mood="neutral",
            reason="r",
        )
        assert nested.exists()
        entries = _read_entries(nested)
        assert len(entries) == 1


# ── singleton helper ──────────────────────────────────────────────────────────


class TestGetNlpErrorLogger:
    def test_returns_nlp_error_logger_instance(self):
        instance = get_nlp_error_logger()
        assert isinstance(instance, NLPErrorLogger)

    def test_singleton_returns_same_object(self):
        a = get_nlp_error_logger()
        b = get_nlp_error_logger()
        assert a is b
