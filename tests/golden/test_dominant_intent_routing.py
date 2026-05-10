from ai.runtime.alice_contract_factory import build_runtime_boundaries
from ai.runtime.contract_pipeline import ContractPipeline
from tests.integration.test_contract_pipeline import _FakeAlice


def test_weather_context_does_not_dominate_alice_objective():
    alice = _FakeAlice()
    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn(
        user_input="doing pretty good, weather is a bit all over the place today, took a nap so i am ready to work on alice",
        user_id="u1",
        turn_number=1,
    )
    assert result.metadata["intent"] != "weather:current"
    assert result.metadata["intent"] in {
        "conversation:goal_statement",
        "operator:continue",
        "operator:next_step",
        "operator:project_status",
    }


def test_codebase_analysis_ability_routes_to_code_request():
    alice = _FakeAlice()
    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn(
        user_input="are you able to analyze her current code base?",
        user_id="u1",
        turn_number=2,
    )
    assert result.metadata["intent"] in {"code:request", "code:list_files"}
    assert "what would you like to start with" not in result.response_text.lower()
