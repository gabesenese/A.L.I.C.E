from ai.contracts import VerifierResult
from ai.runtime.alice_contract_factory import build_runtime_boundaries
from ai.runtime.contract_pipeline import ContractPipeline
from tests.integration.test_contract_pipeline import _FakeAlice


def _pipeline_with_sequence(sequence: list[str]) -> ContractPipeline:
    alice = _FakeAlice()
    responses = iter(list(sequence))

    def _seq_chat(*args, **kwargs):
        try:
            return next(responses)
        except StopIteration:
            return ""

    alice.llm.chat = _seq_chat
    return ContractPipeline(build_runtime_boundaries(alice))


def test_clear_breakdown_retries_and_avoids_generic_clarification():
    pipeline = _pipeline_with_sequence(
        [
            "A proactive companion should observe and suggest useful actions.",
            "I can help. What exact result do you want?",
            "I can help. What exact result do you want?",
        ]
    )
    original_verify = pipeline.boundaries.verifier.verify

    def _verify_with_reject(req):
        if "break this down" in str(req.user_input or "").lower():
            return VerifierResult(
                accepted=False,
                reason="unsupported_claims",
                confidence=0.2,
                diagnostics={},
            )
        return original_verify(req)

    pipeline.boundaries.verifier.verify = _verify_with_reject

    pipeline.run_turn(
        user_input="i dont want it to be like an assistant or chatbot",
        user_id="u_crf1",
        turn_number=1,
    )
    result = pipeline.run_turn(
        user_input="break this down with todays technology",
        user_id="u_crf1",
        turn_number=2,
    )

    low = str(result.response_text or "").lower()
    assert "what exact result do you want" not in low
    assert "model brain" in low
    assert "memory" in low
    assert "tools" in low
    assert "background event loop" in low
    assert "relevance filter" in low
    assert "approval layer" in low


def test_short_followup_uses_active_concept_without_generic_clarification():
    pipeline = _pipeline_with_sequence(
        [
            "A proactive companion keeps context and reacts to meaningful change.",
            "It would look like a layered system with memory, tools, and event monitoring.",
        ]
    )
    pipeline.run_turn(
        user_input="i dont want it to be like an assistant or chatbot",
        user_id="u_crf2",
        turn_number=1,
    )
    result = pipeline.run_turn(
        user_input="what would that look like",
        user_id="u_crf2",
        turn_number=2,
    )

    low = str(result.response_text or "").lower()
    assert "what exact result do you want" not in low
    assert "memory" in low or "layer" in low or "monitoring" in low


def test_truly_ambiguous_without_active_concept_can_clarify():
    pipeline = _pipeline_with_sequence(["I can help. What exact result do you want?"])
    result = pipeline.run_turn(
        user_input="that thing",
        user_id="u_crf3",
        turn_number=1,
    )
    low = str(result.response_text or "").lower()
    assert (
        "what exact result do you want" in low
        or result.metadata.get("intent") == "conversation:clarification_needed"
    )


def test_conceptual_fallback_contains_no_codebase_claims():
    pipeline = _pipeline_with_sequence(
        [
            "Proactivity means Alice should notice change and suggest next actions.",
            "I can help. What exact result do you want?",
            "I can help. What exact result do you want?",
        ]
    )
    original_verify = pipeline.boundaries.verifier.verify

    def _verify_with_reject(req):
        if "todays technology" in str(req.user_input or "").lower():
            return VerifierResult(
                accepted=False,
                reason="unsupported_claims",
                confidence=0.2,
                diagnostics={},
            )
        return original_verify(req)

    pipeline.boundaries.verifier.verify = _verify_with_reject

    pipeline.run_turn(
        user_input="i dont want it to be like an assistant or chatbot",
        user_id="u_crf4",
        turn_number=1,
    )
    result = pipeline.run_turn(
        user_input="break this down with today's technology",
        user_id="u_crf4",
        turn_number=2,
    )

    low = str(result.response_text or "").lower()
    assert ".py" not in low
    assert "ai/runtime" not in low
    assert "i inspected" not in low
    assert "repo" not in low
