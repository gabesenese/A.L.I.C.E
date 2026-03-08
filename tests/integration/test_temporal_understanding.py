"""
tests/integration/test_temporal_understanding.py
─────────────────────────────────────────────────
Unit tests for TemporalUnderstanding — the stable abstraction layer that
wraps TemporalParser and normalises its output into TemporalResult objects.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

from ai.core.nlp_processor import (
    TemporalResult,
    TemporalUnderstanding,
    get_temporal_understanding,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_parser(raw: Optional[Dict[str, Any]]) -> MagicMock:
    """Return a mock TemporalParser whose parse_temporal_expression() returns *raw*."""
    parser = MagicMock()
    parser.parse_temporal_expression.return_value = raw
    return parser


# ── TemporalResult tests ──────────────────────────────────────────────────────

class TestTemporalResult:
    def test_as_dict_contains_required_keys(self):
        tr = TemporalResult(
            start="2025-01-15",
            end=None,
            time=None,
            grain="day",
            raw="January 15th",
            confidence=0.9,
        )
        d = tr.as_dict()
        assert d["date"] == "2025-01-15"
        assert d["end_date"] is None
        assert d["time"] is None
        assert d["grain"] == "day"
        assert d["raw_text"] == "January 15th"
        assert d["confidence"] == 0.9

    def test_as_dict_roundtrips_all_fields(self):
        tr = TemporalResult(
            start="2025-06-01",
            end="2025-06-07",
            time="14:30",
            grain="week",
            raw="next week at 2:30pm",
            confidence=0.85,
        )
        d = tr.as_dict()
        assert d["date"] == "2025-06-01"
        assert d["end_date"] == "2025-06-07"
        assert d["time"] == "14:30"
        assert d["grain"] == "week"
        assert d["confidence"] == 0.85


# ── TemporalUnderstanding.parse() tests ──────────────────────────────────────

class TestTemporalUnderstandingParse:

    def test_returns_none_when_parser_returns_none(self):
        tu = TemporalUnderstanding(_make_parser(None))
        assert tu.parse("no date here") is None

    def test_returns_none_when_parser_returns_empty_dict(self):
        tu = TemporalUnderstanding(_make_parser({}))
        assert tu.parse("some text") is None

    def test_returns_temporal_result_on_success(self):
        raw = {"date": "2025-03-10", "raw_text": "March 10th", "confidence": 0.88}
        tu = TemporalUnderstanding(_make_parser(raw))
        result = tu.parse("March 10th")
        assert isinstance(result, TemporalResult)
        assert result.start == "2025-03-10"
        assert result.raw == "March 10th"
        assert result.confidence == 0.88

    def test_falls_back_to_text_when_raw_text_missing(self):
        raw = {"date": "2025-03-10", "confidence": 0.7}
        tu = TemporalUnderstanding(_make_parser(raw))
        result = tu.parse("March 10th")
        assert result is not None
        assert result.raw == "March 10th"

    def test_defaults_confidence_to_07_when_missing(self):
        raw = {"date": "2025-03-10", "raw_text": "March"}
        tu = TemporalUnderstanding(_make_parser(raw))
        result = tu.parse("March")
        assert result is not None
        assert result.confidence == 0.7

    def test_returns_none_on_parser_exception(self):
        parser = MagicMock()
        parser.parse_temporal_expression.side_effect = RuntimeError("boom")
        tu = TemporalUnderstanding(parser)
        assert tu.parse("tomorrow") is None  # exception is swallowed

    def test_passes_text_to_underlying_parser(self):
        raw = {"date": "2025-01-01", "raw_text": "New Year"}
        parser = _make_parser(raw)
        tu = TemporalUnderstanding(parser)
        tu.parse("New Year")
        parser.parse_temporal_expression.assert_called_once_with("New Year")


# ── _infer_grain() tests ──────────────────────────────────────────────────────

class TestInferGrain:
    """Test the grain inference logic via parse() outputs."""

    def _tu(self, raw: dict) -> TemporalResult:
        """Parse the raw_text field (or 'x') through TemporalUnderstanding."""
        tu = TemporalUnderstanding(_make_parser(raw))
        result = tu.parse(raw.get("raw_text", "x"))
        assert result is not None
        return result

    def test_date_only_gives_day(self):
        # Use a date text that contains no month-name or week-name keywords
        result = self._tu({"date": "2025-04-01", "raw_text": "the 1st"})
        assert result.grain == "day"

    def test_time_only_gives_hour(self):
        result = self._tu({"time": "15:00", "raw_text": "at 3pm"})
        assert result.grain == "hour"

    def test_date_and_time_gives_minute(self):
        result = self._tu({"date": "2025-04-01", "time": "09:30", "raw_text": "April 1st at 09:30"})
        assert result.grain == "minute"

    def test_end_date_with_week_cue_gives_week(self):
        result = self._tu({"date": "2025-04-07", "end_date": "2025-04-13", "raw_text": "this week"})
        assert result.grain == "week"

    def test_end_date_without_week_cue_gives_day(self):
        result = self._tu({"date": "2025-04-07", "end_date": "2025-04-13", "raw_text": "this period"})
        assert result.grain == "day"

    def test_month_cue_gives_month(self):
        result = self._tu({"date": "2025-03-01", "raw_text": "march"})
        assert result.grain == "month"

    def test_week_cue_in_date_gives_week(self):
        result = self._tu({"date": "2025-04-07", "raw_text": "monday"})
        assert result.grain == "week"

    def test_no_temporal_fields_gives_unknown(self):
        # Parser returns truthy dict but no date/time fields
        raw = {"raw_text": "some text", "confidence": 0.3}
        tu = TemporalUnderstanding(_make_parser(raw))
        # With no date/time fields the loop falls through — but parse() returns
        # None for {} and returns a TemporalResult for any truthy dict.
        result = tu.parse("some text")
        assert result is not None
        assert result.grain == "unknown"


# ── factory helper ────────────────────────────────────────────────────────────

class TestGetTemporalUnderstanding:
    def test_returns_temporal_understanding_instance(self):
        parser = MagicMock()
        tu = get_temporal_understanding(parser)
        assert isinstance(tu, TemporalUnderstanding)

    def test_returned_instance_uses_provided_parser(self):
        raw = {"date": "2025-05-01", "raw_text": "May 1st"}
        parser = _make_parser(raw)
        tu = get_temporal_understanding(parser)
        result = tu.parse("May 1st")
        assert result is not None
        assert result.start == "2025-05-01"
