from app.main import ALICE
from ai.core.llm_policy import LLMCallType


class _PhrasingNever:
    def can_phrase_myself(self, *_args, **_kwargs):
        return False

    def record_phrasing(self, *_args, **_kwargs):
        return None


class _GatewayStub:
    def __init__(self, response):
        self.response = response
        self.last_call_type = None

    def request(self, *args, **kwargs):
        self.last_call_type = kwargs.get("call_type")

        class _R:
            success = True
            response = ""

        out = _R()
        out.response = self.response
        return out


def test_generate_natural_response_uses_strict_fallback_without_llm():
    alice = ALICE.__new__(ALICE)
    alice.strict_no_llm = True
    alice.phrasing_learner = _PhrasingNever()
    alice._alice_direct_phrase = lambda *_args, **_kwargs: None

    out = alice._generate_natural_response(
        alice_response={"type": "knowledge_answer", "question": "what is it"},
        tone="helpful",
        context=None,
        user_input="what is it",
    )

    assert out == "I can answer in strict mode, but I need a more specific target question."


def test_alice_direct_phrase_handles_clarification_type():
    alice = ALICE.__new__(ALICE)

    clarify = alice._alice_direct_phrase(
        "clarification_prompt",
        {"options": ["delete one note", "delete all notes"]},
    )

    assert "Do you mean" in clarify


def test_generate_natural_response_uses_scoped_phrase_mode_and_clamp():
    alice = ALICE.__new__(ALICE)
    alice.strict_no_llm = False
    alice.phrasing_learner = _PhrasingNever()
    alice._think = lambda *_args, **_kwargs: None
    alice._last_policy = None
    alice._last_perception = None
    alice._alice_direct_phrase = lambda *_args, **_kwargs: None

    gateway = _GatewayStub(
        "Of course, as an AI language model, I can definitely help with that by giving an extremely long answer "
        + ("x" * 500)
    )
    alice.llm_gateway = gateway

    out = alice._generate_natural_response(
        alice_response={
            "type": "general_response",
            "content": "Please provide a structured and concise response to this payload.",
        },
        tone="professional and precise",
        context=None,
        user_input="help",
    )

    assert gateway.last_call_type in {
        LLMCallType.PHRASE_MICRO,
        LLMCallType.PHRASE_STRUCTURED,
    }
    # The clamp removes the filler opener and the disclaimer clause and keeps the
    # answer. This used to assert a 220-character cap, which only held because the
    # whole reply was being swapped for a short stock line.
    assert "as an ai" not in out.lower()
    assert out.startswith("I can definitely help with that")
