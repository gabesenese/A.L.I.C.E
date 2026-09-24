"""Questions about earlier today, a few days ago or this week get the right period.

requested_period understood dates, "yesterday", "last week" and month names, but
not the ways people usually point at recent conversation: "this morning", "an
hour ago", "earlier", "last night", "three days ago", "this week". Those returned
None, so no time filter applied, and "what did I say this morning?" could be
answered from last month.
"""

from datetime import date

import pytest

from ai.memory.temporal_scope import requested_period

TODAY = date(2026, 9, 24)  # a Thursday
YESTERDAY = date(2026, 9, 23)


@pytest.mark.parametrize(
    "question",
    [
        "what did I say this morning?",
        "do you remember what I said an hour ago?",
        "what did I tell you 20 minutes ago?",
        "what did we talk about earlier today?",
        "what was I saying earlier?",
    ],
)
def test_recent_questions_mean_today(question):
    period = requested_period(question, today=TODAY)
    assert period is not None, question
    assert period.contains(TODAY) and not period.contains(YESTERDAY)


def test_last_night_means_yesterday():
    period = requested_period("what did we discuss last night?", today=TODAY)
    assert period is not None
    assert period.contains(YESTERDAY) and not period.contains(TODAY)


def test_a_number_of_days_ago():
    period = requested_period("what did I tell you three days ago?", today=TODAY)
    assert period is not None
    assert period.contains(date(2026, 9, 21)) and not period.contains(date(2026, 9, 22))


def test_this_week_starts_on_monday():
    period = requested_period("what did we talk about this week?", today=TODAY)
    assert period is not None
    assert period.contains(date(2026, 9, 21)) and period.contains(TODAY)
    assert not period.contains(date(2026, 9, 20))


def test_earlier_this_year_still_means_the_year():
    period = requested_period("what did I say earlier this year?", today=TODAY)
    assert period is not None and period.contains(date(2026, 3, 1))
