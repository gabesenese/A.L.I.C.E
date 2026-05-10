from ai.runtime.alice_contract_factory import build_runtime_boundaries
from ai.runtime.contract_pipeline import ContractPipeline
from tests.integration.test_contract_pipeline import _FakeAlice


def test_research_beginner_request_does_not_clarify():
    alice = _FakeAlice()
    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn(
        user_input="the idea is to create an agentic ai companion, let's do some research how on the technology",
        user_id="u1",
        turn_number=1,
    )
    assert result.metadata["intent"] != "conversation:clarification_needed"
    assert "memory" in result.response_text.lower()
    assert "goals" in result.response_text.lower()
    assert "tools" in result.response_text.lower()
    assert "loop" in result.response_text.lower()


def test_beginner_simple_explanation_is_direct():
    alice = _FakeAlice()
    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn(
        user_input="not really, i am a beginner to this agentic ai companion so i want to learn something simple to understand how it works",
        user_id="u1",
        turn_number=2,
    )
    low = result.response_text.lower()
    assert result.metadata["intent"] == "conversation:educational_explain"
    assert "which one sounds like a good starting point" not in low
    assert "what would you like to start with" not in low
    assert "memory" in low and "goals" in low and "tools" in low and "loop" in low


def test_just_give_me_something_does_not_ask_again():
    alice = _FakeAlice()
    pipeline = ContractPipeline(build_runtime_boundaries(alice))
    pipeline.run_turn(
        user_input="the idea is to create an agentic ai companion, let's do some research how on the technology",
        user_id="u1",
        turn_number=3,
    )
    result = pipeline.run_turn(
        user_input="just give me something, doesnt matter what",
        user_id="u1",
        turn_number=4,
    )
    low = result.response_text.lower()
    assert result.metadata["intent"] == "conversation:educational_explain"
    assert "if that sounds interesting" not in low
    assert "would you like me" not in low
    assert not low.strip().endswith("?")


def test_clarification_still_allowed_for_missing_file_target():
    alice = _FakeAlice()
    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn(
        user_input="read a file",
        user_id="u1",
        turn_number=5,
    )
    assert result.metadata["intent"] in {"code:request", "code:list_files", "file_operations:read"}
