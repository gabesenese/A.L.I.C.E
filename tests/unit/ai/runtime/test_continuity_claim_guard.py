from datetime import datetime, timedelta, timezone

from ai.runtime.continuity_claim_guard import assess_continuity_claims


def test_a_active_objective_without_overlap_does_not_support_claim():
    operator_state = {
        "active_objective": "Improve Alice into an agentic companion/operator",
        "current_focus": "routing",
    }
    result = assess_continuity_claims(
        text="we were discussing machine learning last time",
        memory_items=[],
        operator_state=operator_state,
    )
    assert result.unsupported_continuity_claim is True
    assert "we were discussing machine learning last time" not in result.text.lower()


def test_b_active_objective_with_overlap_supports_claim():
    operator_state = {
        "active_objective": "Improve Alice into an agentic companion/operator",
        "current_focus": "routing",
    }
    result = assess_continuity_claims(
        text="we were discussing routing last time",
        memory_items=[],
        operator_state=operator_state,
    )
    assert result.unsupported_continuity_claim is False
    assert "we were discussing routing last time" in result.text.lower()


def test_c_structured_memory_overlap_supports_claim():
    items = [
        {
            "content": "Gabriel talked about shopping today.",
            "context": {
                "domain": "personal_life",
                "kind": "conversation_event",
                "scope": "day_to_day",
                "confidence": 0.85,
                "timestamp": "2026-05-06T00:00:00+00:00",
                "source": "structured_memory",
            },
        }
    ]
    result = assess_continuity_claims(
        text="you mentioned shopping today",
        memory_items=items,
        operator_state={},
    )
    assert result.unsupported_continuity_claim is False
    assert "shopping" in result.text.lower()


def test_d_structured_memory_no_overlap_rejects_claim():
    items = [
        {
            "content": "Gabriel talked about shopping today.",
            "context": {
                "domain": "personal_life",
                "kind": "conversation_event",
                "scope": "day_to_day",
                "confidence": 0.85,
                "timestamp": "2026-05-06T00:00:00+00:00",
                "source": "structured_memory",
            },
        }
    ]
    result = assess_continuity_claims(
        text="you mentioned machine learning today",
        memory_items=items,
        operator_state={},
    )
    assert result.unsupported_continuity_claim is True


def test_e_recent_session_overlap_supports_claim():
    recent = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    items = [
        {
            "content": "We were debugging Alice's memory routing.",
            "context": {"source": "session", "turn_index": 3, "timestamp": recent},
        }
    ]
    result = assess_continuity_claims(
        text="we were discussing memory routing",
        memory_items=items,
        operator_state={},
    )
    assert result.unsupported_continuity_claim is False


def test_f_vector_memory_alone_is_not_sufficient():
    items = [
        {
            "content": "We discussed machine learning techniques.",
            "context": {"source": "vector_recall", "timestamp": "2025-01-01T00:00:00+00:00"},
        }
    ]
    result = assess_continuity_claims(
        text="we were discussing machine learning last time",
        memory_items=items,
        operator_state={},
    )
    assert result.unsupported_continuity_claim is True


def test_g_mixed_response_removes_only_unsupported_sentence():
    result = assess_continuity_claims(
        text="Hey. Good to see you. We were discussing machine learning last time. Ready to continue?",
        memory_items=[],
        operator_state={},
    )
    low = result.text.lower()
    assert "good to see you" in low
    assert "ready to continue" in low
    assert "machine learning" not in low

