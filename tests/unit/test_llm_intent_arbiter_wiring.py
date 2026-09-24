from types import SimpleNamespace

from app.main import ALICE


class _Nlp:
    def __init__(self):
        self.gateway = None

    def attach_llm_gateway(self, gateway):
        self.gateway = gateway


def _alice(**attrs):
    alice = ALICE.__new__(ALICE)
    for key, value in attrs.items():
        setattr(alice, key, value)
    return alice


def test_the_model_is_attached_as_the_intent_tie_breaker():
    gateway = SimpleNamespace()
    alice = _alice(nlp=_Nlp(), llm_gateway=gateway, strict_no_llm=False)

    alice._wire_llm_intent_arbiter()

    assert alice.nlp.gateway is gateway


def test_strict_policy_keeps_routing_model_free():
    alice = _alice(nlp=_Nlp(), llm_gateway=SimpleNamespace(), strict_no_llm=True)

    alice._wire_llm_intent_arbiter()

    assert alice.nlp.gateway is None


def test_a_vague_message_costs_one_model_call_not_three():
    from ai.core.llm_intent_classifier import LLMIntentClassifier

    calls = []

    class _Gateway:
        def request(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(success=True, response="INTENT: conversation\nCONFIDENCE: 0.8")

    LLMIntentClassifier(_Gateway()).classify_hybrid("hmm that thing", 0.3, "conversation:general")

    assert len(calls) == 1
