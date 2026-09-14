"""Answering a question about a date with a memory from another date.

Recall ranks by similarity and recency. Neither understands "on the 3rd of March
last year", so Alice returned her most relevant memories under a heading that
read as an answer. Telling someone what they said on a day they did not say it
is indistinguishable from remembering, which is what makes it the worst failure
a memory system has.
"""

from datetime import date

import pytest

from ai.memory.temporal_scope import (
    Period,
    any_item_within,
    items_within,
    requested_period,
)

TODAY = date(2026, 9, 14)


def _period(question):
    return requested_period(question, today=TODAY)


# -- what counts as asking about a period ------------------------------------


@pytest.mark.parametrize(
    "question,start,end",
    [
        ("what did I say to you on the 3rd of March last year?", date(2025, 3, 3), date(2025, 3, 3)),
        ("did I tell you anything on March 3rd 2024?", date(2024, 3, 3), date(2024, 3, 3)),
        ("what did we discuss on 2025-07-19?", date(2025, 7, 19), date(2025, 7, 19)),
        ("did I say anything yesterday?", date(2026, 9, 13), date(2026, 9, 13)),
        ("what did we talk about in June?", date(2026, 6, 1), date(2026, 6, 30)),
        ("what did we discuss in December?", date(2026, 12, 1), date(2026, 12, 31)),
        ("remember what I told you in 2024?", date(2024, 1, 1), date(2024, 12, 31)),
        ("what did we talk about last month?", date(2026, 8, 1), date(2026, 8, 31)),
    ],
)
def test_a_named_period_is_recognised(question, start, end):
    period = _period(question)
    assert period is not None, question
    assert (period.start, period.end) == (start, end)


@pytest.mark.parametrize(
    "question",
    [
        "what do you know about how I like answers?",
        "what's the weather in March?",
        "what happened in 1999 in history?",
        "tell me about python",
        "",
    ],
)
def test_questions_that_do_not_ask_about_a_remembered_period_are_left_alone(question):
    """An unrecognised phrasing must change no behaviour, so the guard only
    fires where the constraint is unambiguous."""
    assert _period(question) is None


@pytest.mark.parametrize(
    "question",
    [
        "did I say anything on the 31st of February?",
        "what did I tell you on the 99th of March?",
        "did we discuss anything on 2025-13-45?",
    ],
)
def test_an_impossible_date_falls_back_rather_than_raising(question):
    """A date that cannot exist must not crash the turn. Returning the wider
    period, or nothing, both leave the caller behaving as it did before."""
    period = _period(question)
    if period is not None:
        assert period.start <= period.end


# -- whether a memory falls inside it ----------------------------------------


def _item(timestamp, content="something"):
    return {"timestamp": timestamp, "content": content}


def test_a_memory_inside_the_period_counts():
    period = Period(date(2025, 3, 1), date(2025, 3, 31), "March 2025")
    assert any_item_within([_item("2025-03-14T09:00:00")], period) is True


def test_a_memory_outside_the_period_does_not():
    period = Period(date(2025, 3, 1), date(2025, 3, 31), "March 2025")
    assert any_item_within([_item("2026-05-02T09:00:00")], period) is False


def test_a_memory_with_no_timestamp_cannot_support_a_claim_about_when():
    period = Period(date(2025, 3, 1), date(2025, 3, 31), "March 2025")
    assert any_item_within([{"content": "no timestamp at all"}], period) is False


def test_an_unparseable_timestamp_does_not_count_or_raise():
    period = Period(date(2025, 3, 1), date(2025, 3, 31), "March 2025")
    assert any_item_within([_item("some time last spring")], period) is False


def test_a_timestamp_nested_in_context_is_read():
    period = Period(date(2025, 3, 1), date(2025, 3, 31), "March 2025")
    item = {"content": "x", "context": {"timestamp": "2025-03-09T12:00:00"}}
    assert any_item_within([item], period) is True


def test_items_within_keeps_only_the_matching_memories():
    period = Period(date(2025, 3, 1), date(2025, 3, 31), "March 2025")
    items = [
        _item("2026-05-02T09:00:00", "unrelated and recent"),
        _item("2025-03-14T09:00:00", "the one actually from then"),
        _item("2025-03-20T09:00:00", "also from then"),
    ]
    kept = items_within(items, period)
    assert [i["content"] for i in kept] == ["the one actually from then", "also from then"]


def test_boundaries_are_inclusive():
    period = Period(date(2025, 3, 1), date(2025, 3, 31), "March 2025")
    assert any_item_within([_item("2025-03-01T00:00:01")], period) is True
    assert any_item_within([_item("2025-03-31T23:59:59")], period) is True
    assert any_item_within([_item("2025-04-01T00:00:01")], period) is False


def test_a_timezone_aware_timestamp_is_handled():
    period = Period(date(2025, 3, 1), date(2025, 3, 31), "March 2025")
    assert any_item_within([_item("2025-03-14T09:00:00Z")], period) is True


def test_describe_reads_as_a_person_would_say_it():
    assert Period(date(2025, 3, 3), date(2025, 3, 3), "x").describe() == "3 March 2025"
    assert Period(date(2025, 3, 1), date(2025, 3, 31), "x").describe() == "March 2025"
    assert Period(date(2024, 1, 1), date(2024, 12, 31), "x").describe() == "2024"
