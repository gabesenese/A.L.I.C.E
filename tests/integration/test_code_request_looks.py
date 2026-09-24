"""A code request nothing handled is answered, not met with a promise to go and look."""

from ai.contracts import MemoryResult, RouterDecision
from ai.contracts.runtime_contracts import ResponseRequest
from ai.runtime.alice_contract_factory import build_runtime_boundaries
from tests.integration.test_contract_pipeline import _FakeAlice


def test_unhandled_code_request_does_not_promise_to_look():
    alice = _FakeAlice()
    alice._handle_code_request = lambda *args, **kwargs: ""
    boundaries = build_runtime_boundaries(alice)

    output = boundaries.response.generate(
        ResponseRequest(
            user_input="can you look at my code?",
            decision=RouterDecision(route="llm", intent="code:request", confidence=0.9, decision_band="execute"),
            memory=MemoryResult(items=[]),
        )
    )

    assert "I will start by listing" not in output.text
    assert output.metadata.get("type") != "code_request_fallback"
    assert output.text.startswith("LLM:")
