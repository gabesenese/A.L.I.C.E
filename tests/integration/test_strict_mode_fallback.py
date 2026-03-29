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
        alice_response={"type": "operation_success", "operation": "run_tests"},
        tone="helpful",
        context=None,
        user_input="run tests",
    )

    assert out == "Completed: run tests."


def test_alice_direct_phrase_handles_operation_and_clarification_types():
    alice = ALICE.__new__(ALICE)

    ok = alice._alice_direct_phrase(
        "operation_success",
        {"operation": "delete_note", "details": {"title": "Shopping"}},
    )
    fail = alice._alice_direct_phrase(
        "operation_failure",
        {"operation": "delete_note", "error": "note not found"},
    )
    clarify = alice._alice_direct_phrase(
        "clarification_prompt",
        {"options": ["delete one note", "delete all notes"]},
    )

    assert ok == "Done: delete note for 'Shopping'."
    assert "I couldn't complete delete note" in fail
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

    assert gateway.last_call_type in {LLMCallType.PHRASE_MICRO, LLMCallType.PHRASE_STRUCTURED}
    assert "as an ai" not in out.lower()
    assert len(out) <= 220
