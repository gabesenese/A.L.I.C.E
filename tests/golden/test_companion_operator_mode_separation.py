from ai.runtime.alice_contract_factory import build_runtime_boundaries
from ai.runtime.contract_pipeline import ContractPipeline
from tests.integration.test_contract_pipeline import _FakeAlice


def test_casual_how_are_you_does_not_inject_project_status():
    alice = _FakeAlice()
    alice._operator_state = {
        "active_objective": "Improve Alice into an agentic companion/operator",
        "current_focus": "agent_loop.py",
        "next_recommended_action": "inspect ai/runtime/agent_loop.py",
    }
    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn(
        user_input="how are you?",
        user_id="u1",
        turn_number=1,
    )
    low = result.response_text.lower()
    assert "current objective is" not in low
    assert "next best move" not in low
    assert "current focus" not in low
    assert "agent_loop.py" not in low
    assert "processing some interesting stuff in the background" not in low
    assert "working behind the scenes" not in low


def test_continue_turn_can_include_next_move():
    alice = _FakeAlice()
    alice._operator_state = {
        "active_objective": "Improve Alice into an agentic companion/operator",
        "current_focus": "agent_loop.py",
        "next_recommended_action": "inspect ai/runtime/agent_loop.py",
    }
    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn(
        user_input="what's next?",
        user_id="u1",
        turn_number=2,
    )
    low = result.response_text.lower()
    assert "next best move" in low
