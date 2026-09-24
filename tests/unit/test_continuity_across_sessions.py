"""After a restart she can say what you were doing last time, and only that.

The transcript survives a restart (test_transcript_survives_restart), so the
model sees the previous session and says "last time we talked about the
tokenizer". The continuity guard deleted that sentence: a claim about an earlier
occasion needed stored memory, and the restored transcript did not count.
"""

from ai.runtime.continuity_claim_guard import assess_continuity_claims

LAST_SESSION = "the parser is slow, I think the tokenizer backtracks"


def _assess(text, *, last_session="", this_turn=""):
    return assess_continuity_claims(
        text=text,
        memory_items=[],
        operator_state={},
        evidence_text=this_turn,
        prior_session_text=last_session,
    )


def test_last_time_is_grounded_by_the_previous_session():
    result = _assess(
        "Last time we talked about the tokenizer in your parser. Is it still slow?",
        last_session=LAST_SESSION,
    )
    assert result.unsupported_continuity_claim is False
    assert "tokenizer" in result.text


def test_last_time_about_something_else_is_still_removed():
    result = _assess(
        "Last time we talked about your sister's wedding. How did it go?",
        last_session=LAST_SESSION,
    )
    assert result.unsupported_continuity_claim is True
    assert "wedding" not in result.text


def test_this_session_still_does_not_prove_an_earlier_one():
    result = _assess("Last time we talked about the tokenizer.", this_turn="what about the tokenizer")
    assert result.unsupported_continuity_claim is True


def test_a_name_he_used_last_session_is_not_an_invention():
    result = _assess(
        "Are you still meeting Priya about the parser?",
        last_session="I'm meeting Priya on Friday about the parser rewrite",
    )
    assert result.unsupported_continuity_claim is False
    assert "Priya" in result.text
