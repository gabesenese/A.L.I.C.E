from ai.runtime.alice_contract_factory import build_runtime_boundaries
from ai.runtime.contract_pipeline import ContractPipeline
from tests.integration.test_contract_pipeline import _FakeAlice


def test_stores_active_concept_thread_for_proactive_companion():
    alice = _FakeAlice()
    pipeline = ContractPipeline(build_runtime_boundaries(alice))
    result = pipeline.run_turn(
        user_input="i dont want it to be like an assistant or chatbot",
        user_id="u_ct1",
        turn_number=1,
    )
    assert result.metadata.get("intent") == "conversation:concept_refinement"
    thread = dict(getattr(alice, "_active_concept_thread", {}) or {})
    assert str(thread.get("topic") or "") == "proactive AI companion"
    constraints = [str(x).lower() for x in list(thread.get("constraints") or [])]
    assert any("not assistant" == c for c in constraints)
    assert any("not chatbot" == c for c in constraints)


def test_resolves_like_this_to_active_concept():
    alice = _FakeAlice()
    pipeline = ContractPipeline(build_runtime_boundaries(alice))
    pipeline.run_turn(
        user_input="i dont want it to be like an assistant or chatbot",
        user_id="u_ct2",
        turn_number=1,
    )
    result = pipeline.run_turn(
        user_input="i want alice to be proactive, like this",
        user_id="u_ct2",
        turn_number=2,
    )
    assert result.metadata.get("intent") == "conversation:concept_refinement"
    assert result.metadata.get("route") == "llm"


def test_resolves_something_like_jarvis_to_active_concept():
    alice = _FakeAlice()
    pipeline = ContractPipeline(build_runtime_boundaries(alice))
    pipeline.run_turn(
        user_input="i want alice to be proactive",
        user_id="u_ct3",
        turn_number=1,
    )
    result = pipeline.run_turn(
        user_input="something like jarvis",
        user_id="u_ct3",
        turn_number=2,
    )
    assert result.metadata.get("intent") == "conversation:concept_refinement"
    assert result.metadata.get("route") == "llm"


def test_concept_refinement_not_treated_as_codebase_claim():
    alice = _FakeAlice()
    alice.llm.chat = (
        lambda *args, **kwargs: "Actual proactivity means a runtime loop that observes events, detects change, scores relevance, and suggests action."
    )
    pipeline = ContractPipeline(build_runtime_boundaries(alice))
    result = pipeline.run_turn(
        user_input="i want it to be actually proactive",
        user_id="u_ct4",
        turn_number=1,
    )
    low = str(result.response_text or "").lower()
    assert "i have not verified the codebase yet" not in low
    assert "runtime loop" in low


def test_implementation_request_separate_from_concept_refinement():
    alice = _FakeAlice()
    pipeline = ContractPipeline(build_runtime_boundaries(alice))
    result = pipeline.run_turn(
        user_input="how do we implement this in Alice?",
        user_id="u_ct5",
        turn_number=1,
    )
    assert result.metadata.get("route") == "local"
    assert result.metadata.get("intent") in {"code:request", "operator:continue", "code:list_files"}
