"""Asked to read a file, she says what it does, not how many lines it has."""

from ai.contracts import MemoryResult, RouterDecision, ToolResult
from ai.contracts.runtime_contracts import ResponseRequest
from ai.runtime.alice_contract_factory import build_runtime_boundaries
from ai.runtime.local_actions.local_action_executor import _excerpt
from tests.integration.test_contract_pipeline import _FakeAlice

_STATS = "Inspected `ai/runtime/agent_loop.py`. Structural stats: 419 lines, 5 imports, 4 classes, 12 functions."
_SOURCE = "class AgentLoop:\n    '''Runs plan, act, observe until the goal is met or 3 steps pass.'''\n"


class _ReadingLlm:
    def __init__(self, reply):
        self.reply = reply
        self.contexts = []

    def chat(self, user_input, use_history=True, **kwargs):
        self.contexts.append(str(kwargs.get("context") or ""))
        return self.reply


def _read(reply):
    alice = _FakeAlice()
    alice.llm = _ReadingLlm(reply)
    output = build_runtime_boundaries(alice).response.generate(
        ResponseRequest(
            user_input="read ai/runtime/agent_loop.py",
            decision=RouterDecision(route="local", intent="code:request", confidence=0.9, decision_band="execute"),
            memory=MemoryResult(items=[]),
            tool_result=ToolResult(
                success=True,
                tool_name="local_action_executor",
                action="code:analyze_file",
                data={
                    "response": _STATS,
                    "file_text": _SOURCE,
                    "local_execution": {"inspected_file": "ai/runtime/agent_loop.py"},
                },
            ),
        )
    )
    return alice, output


def test_the_reply_is_written_from_the_source():
    alice, output = _read("It runs the plan-act-observe loop, giving up after 3 steps.")

    assert output.text == "It runs the plan-act-observe loop, giving up after 3 steps."
    assert any("plan, act, observe" in c for c in alice.llm.contexts)


def test_a_made_up_number_falls_back_to_the_inspection():
    _, output = _read("It runs the agent loop for up to 40 steps.")

    assert "419 lines" in output.text


def test_long_files_keep_their_head_and_tail():
    text = "HEAD" + "x" * 20000 + "TAIL"
    excerpt = _excerpt(text)
    assert excerpt.startswith("HEAD") and excerpt.endswith("TAIL") and len(excerpt) < 6100
