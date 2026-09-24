"""A yes answers the question it follows.

A code recommendation is stored on nearly every turn, and "yes" at any point ran
the last one. Asked "want me to remind you about it tonight?", a "yes" came back
as an inspection of ai/runtime/agent_loop.py.
"""

from ai.memory.project_memory import ProjectMemoryState, save_project_state
from ai.runtime.alice_contract_factory import build_runtime_boundaries
from ai.runtime.contract_pipeline import ContractPipeline
from tests.integration.test_contract_pipeline import _FakeAlice

RECOMMENDATION = {
    "action": "inspect_file",
    "target": "ai/runtime/agent_loop.py",
    "reason": "Active objective exists; agent loop should drive next safe step.",
    "safety_level": "safe_read",
    "requires_approval": False,
    "source": "next_step_policy",
}


def _reply_to(last_reply, text="yes"):
    save_project_state(
        ProjectMemoryState(
            active_objective="Improve Alice into an agentic companion/operator",
            last_recommended_action=dict(RECOMMENDATION),
        ),
        user_id="default",
    )
    alice = _FakeAlice()
    alice.llm.conversation_history = [
        {"role": "user", "content": "I need to call the bank later"},
        {"role": "assistant", "content": last_reply},
    ]
    return ContractPipeline(build_runtime_boundaries(alice)).run_turn(user_input=text, user_id="default", turn_number=2)


def test_yes_to_something_else_does_not_run_a_stored_recommendation():
    result = _reply_to("Want me to remind you about it tonight?")

    assert result.metadata["intent"] != "operator:execute_recommended_action"
    assert "agent_loop" not in result.response_text


def test_yes_to_the_recommendation_itself_runs_it():
    result = _reply_to("Next I'd look at agent_loop.py. Want me to go through it?", "go ahead")

    assert result.metadata["intent"] == "operator:execute_recommended_action"
