"""A tool turn sees the same memory and context as a conversational one."""

from ai.contracts import MemoryResult, RouterDecision
from ai.contracts.runtime_contracts import ResponseRequest
from ai.runtime.alice_contract_factory import build_runtime_boundaries
from tests.integration.test_contract_pipeline import _FakeAlice, _ToolLlm


def test_the_tool_loop_is_given_the_companion_context():
    alice = _FakeAlice()
    alice.llm = _ToolLlm()

    output = build_runtime_boundaries(alice).response.generate(
        ResponseRequest(
            user_input="which files are in the ai folder?",
            decision=RouterDecision(
                route="llm", intent="conversation:general", confidence=0.9, decision_band="execute"
            ),
            memory=MemoryResult(
                items=[{"content": "User said: we ship the runtime refactor on Friday\nAlice replied: Noted."}]
            ),
        )
    )

    first_request = alice.llm.tool_calls_seen[0]
    system_text = " ".join(m["content"] for m in first_request if m.get("role") == "system")
    assert "we ship the runtime refactor on Friday" in system_text
    assert "runtime and memory packages" in output.text
