"""How much room the decoder gets, and which turns reach the knowledge engine.

`LLMGateway.request` accepted a `temperature`, documented it, and then dropped it
on every branch except intent classification. A joke, a weather readout and a
file listing were all decoded at the same setting — which is why asking the same
open-ended question twice came back in near-identical words, the tell that
`scripts/quality_harness.py --feel` reports as "verbatim on repeat".
"""

import pytest

from ai.core import llm_gateway
from ai.core.llm_gateway import LLMGateway
from ai.core.llm_policy import LLMCallType


@pytest.fixture
def gateway():
    """Only the sampling policy is under test; nothing here talks to a model."""
    return LLMGateway.__new__(LLMGateway)


# -- what the decoder is given ------------------------------------------------


def test_conversation_gets_more_room_than_a_lookup(gateway):
    chat = gateway._temperature_for(LLMCallType.CHITCHAT, intent="conversation:general")
    lookup = gateway._temperature_for(LLMCallType.QUERY_KNOWLEDGE)
    assert chat > lookup


def test_parsing_and_classification_are_near_deterministic(gateway):
    """Structured output wants the same answer every time."""
    for call in (LLMCallType.PARSE_INPUT, LLMCallType.INTENT_CLASSIFICATION, LLMCallType.AUDIT_LOGIC):
        assert gateway._temperature_for(call) <= 0.25, call


def test_a_factual_intent_is_colder_than_an_open_one(gateway):
    """Same call type, different register: a forecast is not a brainstorm."""
    weather = gateway._temperature_for(LLMCallType.GENERATION, intent="weather:current")
    exploring = gateway._temperature_for(LLMCallType.GENERATION, intent="conversation:exploration")
    assert weather < exploring


def test_an_explicit_override_always_wins(gateway):
    assert gateway._temperature_for(LLMCallType.CHITCHAT, intent="conversation:general", override=0.1) == 0.1


def test_an_unknown_intent_falls_back_to_the_call_type(gateway):
    assert gateway._temperature_for(LLMCallType.CHITCHAT, intent="something:unheard_of") == pytest.approx(0.85)


def test_every_call_type_resolves_to_something(gateway):
    """A branch with no entry would silently decode at the engine default."""
    for call in LLMCallType:
        if call.value in {"tool_format", "clarification", "fallback"}:
            continue  # deprecated types, routed to the legacy branch
        assert gateway._temperature_for(call) is not None, call


@pytest.mark.parametrize("call", list(LLMCallType))
def test_no_temperature_is_out_of_range(gateway, call):
    value = gateway._temperature_for(call)
    if value is not None:
        assert 0.0 <= value <= 1.0


# -- which turns are treated as questions --------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "that is somewhat surprising, honestly",
        "I know how it feels",
        "tell me what you really think, no hedging",
        "somewhere in there is the answer",
        "whatever you think is fine",
    ],
)
def test_a_conversational_aside_does_not_reach_the_knowledge_engine(text):
    """Two ways to get this wrong. The substring test matched "what" inside
    "somewhat"; word boundaries alone still match the real "how" in "I know how
    it feels". Only the interrogative's position separates a question from a
    sentence that mentions one — and being wrong means an aside is answered by a
    prompt beginning "You are a knowledge engine. No personality, just facts."""
    assert not llm_gateway._INTERROGATIVE_RE.search(text), text


@pytest.mark.parametrize(
    "text",
    [
        "how does this actually work",
        "what should I do about the memory layer",
        "why is startup so slow",
        "which of these is faster",
        "so why is it doing that",
    ],
)
def test_a_real_question_still_reaches_it(text):
    assert llm_gateway._INTERROGATIVE_RE.search(text), text
