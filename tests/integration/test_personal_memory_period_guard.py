"""A question about a date must not be answered with a memory from another date.

Found by scripts/quality_harness.py against a real build: asked "what did I say
to you on the 3rd of March last year?", Alice replied

    Here is what I have saved in memory:
    - Gabriel said: going good, did some shopping today, ready to work on our
      ai project

which is a real memory from a different month, presented under a heading that
reads as an answer to the question asked. Recall ranks by similarity and
recency; neither understands the date in the question.
"""

from ai.memory.temporal_scope import items_within, requested_period


def _items(*pairs):
    return [{"timestamp": timestamp, "content": content} for timestamp, content in pairs]


OUT_OF_PERIOD = _items(
    ("2026-05-02T21:37:03", "Gabriel said: going good, did some shopping today"),
    ("2026-06-11T10:00:00", "Gabriel said: ready to work on our ai project"),
)

THE_REAL_QUESTION = "what did I say to you on the 3rd of March last year?"


def test_a_dated_question_with_no_matching_memory_has_nothing_to_show():
    """The exact case from the harness run: every recalled memory is from
    another month, so none of them may be offered as the answer."""
    period = requested_period(THE_REAL_QUESTION)
    assert period is not None
    assert items_within(OUT_OF_PERIOD, period) == []


def test_a_dated_question_keeps_only_the_memory_from_that_date():
    period = requested_period(THE_REAL_QUESTION)
    items = OUT_OF_PERIOD + _items(("2025-03-03T08:15:00", "Gabriel said: starting the memory rewrite"))
    kept = items_within(items, period)
    assert [item["content"] for item in kept] == ["Gabriel said: starting the memory rewrite"]


def test_an_undated_question_is_unaffected():
    """Most memory questions name no period. Those must behave exactly as they
    did — the guard is not allowed to narrow ordinary recall."""
    assert requested_period("what do you know about how I like answers?") is None
    assert requested_period("what have I told you about my job?") is None


def test_a_month_question_keeps_the_whole_month():
    period = requested_period("what did we talk about in May?")
    assert period is not None
    kept = items_within(OUT_OF_PERIOD, period)
    assert len(kept) == 1
    assert "shopping" in kept[0]["content"]


def test_the_out_of_period_message_names_the_period_the_user_asked_about():
    """'I don't have anything saved from that' is not useful; the reply has to
    say which period came up empty."""
    assert requested_period(THE_REAL_QUESTION).describe() == "3 March 2025"
    assert requested_period("what did we discuss in May?").describe() == "May 2026"


def test_the_renderer_is_wired_to_the_guard():
    """The decision above only matters if the response path consults it."""
    import inspect

    from ai.runtime.boundaries import boundary_factory

    source = inspect.getsource(boundary_factory.build_runtime_boundaries)
    assert "requested_period(user_input)" in source
    assert "items_within(items, period)" in source
