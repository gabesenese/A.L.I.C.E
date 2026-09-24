"""When she is unsure, her question names what was unclear in what was said."""

from ai.contracts import MemoryResult, RouterDecision
from ai.contracts.runtime_contracts import ResponseRequest
from ai.runtime.alice_contract_factory import build_runtime_boundaries
from tests.integration.test_contract_pipeline import _FakeAlice


class _AskingLlm:
    def __init__(self, reply):
        self.reply = reply
        self.contexts = []

    def chat(self, user_input, use_history=True, **kwargs):
        self.contexts.append(str(kwargs.get("context") or ""))
        return self.reply


def _clarify(reply):
    alice = _FakeAlice()
    alice.llm = _AskingLlm(reply)
    output = build_runtime_boundaries(alice).response.generate(
        ResponseRequest(
            user_input="fix it",
            decision=RouterDecision(
                route="clarify",
                intent="clarification:context_resolution",
                confidence=0.2,
                decision_band="clarify",
                needs_clarification=True,
                metadata={"pronouns": ["it"], "options": ["the login bug", "the test failure"]},
            ),
            memory=MemoryResult(items=[]),
        )
    )
    return alice, output


def test_the_question_names_the_ambiguity():
    alice, output = _clarify("Do you mean the login bug or the test failure?")

    assert output.text == "Do you mean the login bug or the test failure?"
    assert output.follow_up_question == output.text
    assert any("the login bug" in c for c in alice.llm.contexts)


def test_a_reply_that_is_not_a_question_falls_back():
    _, output = _clarify("I will fix the login bug now.")

    assert output.text == "What exact result should I produce next?"
