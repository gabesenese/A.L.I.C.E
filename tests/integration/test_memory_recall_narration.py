"""'Do you remember...' is answered in her words, checked against the saved rows."""

from ai.memory.personal_memory import PersonalMemoryStore
from ai.runtime.alice_contract_factory import build_runtime_boundaries
from ai.runtime.contract_pipeline import ContractPipeline
from tests.integration.test_contract_pipeline import _FakeAlice


class _RecallLlm:
    def __init__(self, reply):
        self.reply = reply
        self.contexts = []

    def chat(self, user_input, use_history=True, **kwargs):
        self.contexts.append(str(kwargs.get("context") or ""))
        return self.reply


def _ask(reply):
    alice = _FakeAlice()
    alice.llm = _RecallLlm(reply)
    PersonalMemoryStore(alice.memory).store_structured_memory(
        content="You said your sister visited last weekend.",
        domain="personal_life",
        kind="conversation_event",
        scope="day_to_day",
        confidence=0.9,
        source="conversation",
    )
    pipeline = ContractPipeline(build_runtime_boundaries(alice))
    result = pipeline.run_turn(user_input="what did i talk about my personal life?", user_id="u1", turn_number=47)
    return alice, result


def test_recall_is_phrased_by_the_model_from_the_saved_rows():
    alice, result = _ask("You mentioned your sister visited last weekend.")

    assert result.metadata["response_type"] == "personal_memory_grounded"
    assert result.response_text == "You mentioned your sister visited last weekend."
    assert any("sister visited last weekend" in c for c in alice.llm.contexts)


def test_recall_keeps_the_saved_rows_when_the_model_adds_to_them():
    _, result = _ask("Your sister flew in from Lisbon with her two kids and a dog.")

    assert "here is what i have saved in memory" in result.response_text.lower()
    assert "sister visited last weekend" in result.response_text.lower()
