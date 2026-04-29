from ai.runtime.fallback_policy import RuntimeFallbackPolicy
from ai.runtime.response_authority import ResponseAuthorityContract


def test_response_authority_publishes_accepted_llm_response():
    policy = RuntimeFallbackPolicy()
    contract = ResponseAuthorityContract(policy)
    contract.start_turn("t1")

    outcome = contract.resolve_llm_turn(
        accepted=True,
        response="Use a route-execute-verify-respond pipeline.",
        refine_fn=lambda _: "",
        deterministic_fn=lambda: "fallback",
    )

    assert outcome.action == "publish"
    assert "route-execute-verify-respond" in outcome.text
    assert outcome.reason == "llm_accepted"


def test_response_authority_refines_before_deterministic_fallback():
    policy = RuntimeFallbackPolicy()
    contract = ResponseAuthorityContract(policy)
    contract.start_turn("t2")

    calls = {"deterministic": 0}

    outcome = contract.resolve_llm_turn(
        accepted=False,
        response="raw llm draft",
        refine_fn=lambda _: "refined answer",
        deterministic_fn=lambda: calls.__setitem__(
            "deterministic", calls["deterministic"] + 1
        ),
    )

    assert outcome.action == "refine"
    assert outcome.text == "refined answer"
    assert calls["deterministic"] == 0


def test_response_authority_uses_deterministic_when_refine_empty():
    policy = RuntimeFallbackPolicy()
    contract = ResponseAuthorityContract(policy)
    contract.start_turn("t3")

    calls = {"deterministic": 0}

    def _deterministic():
        calls["deterministic"] += 1
        return "deterministic fallback"

    outcome = contract.resolve_llm_turn(
        accepted=False,
        response="raw llm draft",
        refine_fn=lambda _: "",
        deterministic_fn=_deterministic,
    )

    assert outcome.action == "deterministic_fallback"
    assert outcome.text == "deterministic fallback"
    assert calls["deterministic"] == 1
