from ai.runtime.alice_contract_factory import build_runtime_boundaries
from ai.runtime.contract_pipeline import ContractPipeline
from tests.integration.test_contract_pipeline import _FakeAlice


def test_learn_more_about_agentic_ai_routes_educational():
    alice = _FakeAlice()
    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn(
        user_input="i wanna learn more about agentic ai and how we can create it",
        user_id="u1",
        turn_number=1,
    )
    assert result.metadata["intent"] == "conversation:educational_explain"
    assert "NoneType" not in result.response_text
    assert "planner/executor error" not in result.response_text.lower()


def test_learn_basics_of_industry_routes_educational():
    alice = _FakeAlice()
    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn(
        user_input="i actually want to learn the basics of the agentic ai industry",
        user_id="u1",
        turn_number=2,
    )
    assert result.metadata["intent"] == "conversation:educational_explain"


def test_followup_information_uses_active_learning_topic():
    alice = _FakeAlice()
    pipeline = ContractPipeline(build_runtime_boundaries(alice))
    pipeline.run_turn(
        user_input="i actually want to learn the basics of the agentic ai industry",
        user_id="u1",
        turn_number=3,
    )
    result = pipeline.run_turn(
        user_input="yeah give me some information",
        user_id="u1",
        turn_number=4,
    )
    assert result.metadata["intent"] == "conversation:educational_explain"
    assert "clarification" not in result.response_text.lower()
    assert "industry" in result.response_text.lower() or "agentic ai" in result.response_text.lower()


def test_tell_me_more_and_examples_keep_same_topic():
    alice = _FakeAlice()
    pipeline = ContractPipeline(build_runtime_boundaries(alice))
    pipeline.run_turn(
        user_input="i actually want to learn the basics of the agentic ai industry",
        user_id="u1",
        turn_number=5,
    )
    more = pipeline.run_turn(user_input="tell me more", user_id="u1", turn_number=6)
    examples = pipeline.run_turn(user_input="give me examples", user_id="u1", turn_number=7)
    assert more.metadata["intent"] == "conversation:educational_explain"
    assert examples.metadata["intent"] == "conversation:educational_explain"


def test_project_work_still_routes_project_not_educational():
    alice = _FakeAlice()
    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn(
        user_input="i want to work on alice and implement the agentic loop",
        user_id="u1",
        turn_number=8,
    )
    assert result.metadata["intent"] in {"operator:continue", "conversation:goal_statement"}
    assert result.metadata["intent"] != "conversation:educational_explain"


def test_codex_request_still_routes_implementation_path():
    alice = _FakeAlice()
    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn(
        user_input="give me a codex input to implement agentic ai foundations",
        user_id="u1",
        turn_number=9,
    )
    assert result.metadata["intent"] in {"code:request", "code:list_files", "conversation:goal_statement"}
    assert result.metadata["intent"] != "conversation:educational_explain"
