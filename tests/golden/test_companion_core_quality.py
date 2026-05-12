from ai.runtime.claim_verifier import verify_response_claims
from ai.runtime.perception_frame import build_perception_frame
from ai.runtime.greeting_surface_policy import render_grounded_greeting


def test_human_context_plus_action_request():
    frame = build_perception_frame(
        "good, it was a long day, but its just monday so we are trying to stay positive, I want to work on alice for a little bit"
    )
    assert "long day" in frame.social_context.lower()
    assert "work on alice" in frame.actual_request.lower()
    assert frame.is_action_request is True


def test_casual_context_plus_notes_request():
    frame = build_perception_frame(
        "weather is great today, I was wondering if i have any open notes?"
    )
    assert "weather is great today" in frame.social_context.lower()
    assert "open notes" in frame.actual_request.lower()


def test_memory_deletion_honesty_claim_guard():
    out = verify_response_claims("I deleted memories about your mom.", memory_result={})
    assert out.valid is False


def test_fake_action_claim_blocked():
    out = verify_response_claims("I inspected ai/runtime/agent_loop.py.", local_execution={})
    assert out.valid is False


def test_greeting_quality_rejects_generic_empty():
    result = render_grounded_greeting(
        user_name="Gabriel",
        user_input="hi alice",
        llm_generate=lambda *args, **kwargs: "Hey Gabriel! It's great to chat with you!",
    )
    assert "great to chat with you" not in result.text.lower()
