"""An ordinary question is generated once.

Every question that was not small talk went to the model three times: the tool
loop answered it, the answer was thrown away because no tool had been used, the
loop ran again further down, and then chat() generated the reply the user saw.
That is three waits on a local model, and three times the usage on a hosted one.

The loop now opens an ordinary turn with exactly what chat() would send, plus the
tool rules, so an answer that needed no tool is the reply. It still goes through
every check a chat reply does.
"""

from types import SimpleNamespace

import pytest

from ai.core import persona
from ai.core.llm_engine import LLMConfig, LocalLLMEngine
from ai.core.react_loop import SYSTEM_PROMPT
from ai.runtime.alice_contract_factory import build_runtime_boundaries
from ai.runtime.boundaries import boundary_factory
from ai.runtime.contract_pipeline import ContractPipeline
from tests.integration.test_contract_pipeline import _FakeAlice

QUESTION = "should I rewrite the memory layer in rust?"
REPLY = "I'd leave it. You would be rewriting the one part that currently works."


@pytest.fixture
def model(monkeypatch):
    """The real engine with its transport stubbed, keeping every request it sends."""
    engine = LocalLLMEngine(LLMConfig(model="test-model"))
    sent = []
    replies = []
    monkeypatch.setattr(engine, "_ensure_service_probed", lambda: None)
    monkeypatch.setattr(engine, "_available_models", [])

    def post(url, payload, what=""):
        sent.append(payload)
        return {"message": {"content": replies.pop(0) if replies else REPLY}}

    monkeypatch.setattr(engine, "_post_with_retry", post)
    return SimpleNamespace(engine=engine, sent=sent, replies=replies)


def _ask(engine, text=QUESTION):
    alice = _FakeAlice()
    alice.llm = engine
    return ContractPipeline(build_runtime_boundaries(alice)).run_turn(user_input=text, user_id="u1", turn_number=1)


def test_an_ordinary_question_is_generated_once(model):
    result = _ask(model.engine)

    assert len(model.sent) == 1
    assert result.response_text == REPLY


def test_that_generation_sees_what_chat_would_have_sent_plus_the_tools(model):
    _ask(model.engine)

    messages = model.sent[0]["messages"]
    assert messages[0]["content"].startswith(persona.identity())
    # The conversational body, not the narrower one the loop uses for code work.
    assert "The first word carries content" in messages[0]["content"]
    assert messages[1]["content"] == persona.tool_guidance()
    assert messages[-1] == {"role": "user", "content": QUESTION}
    assert model.sent[0]["tools"]


def test_the_exchange_is_in_the_transcript_once(model):
    _ask(model.engine)

    assert [m["content"] for m in model.engine.conversation_history] == [QUESTION, REPLY]


def test_the_reply_still_goes_through_the_chat_checks(model):
    """A hedge is still sent back for a real answer, and the user sees that one."""
    model.replies.extend(["I'm not sure.", REPLY])

    result = _ask(model.engine)

    assert len(model.sent) == 2
    assert result.response_text == REPLY
    assert [m["content"] for m in model.engine.conversation_history] == [QUESTION, REPLY]


def test_a_code_turn_keeps_the_loops_own_prompt(model):
    req = SimpleNamespace(
        user_input="what does the tier file decide?",
        decision=SimpleNamespace(route="local", intent="code:request", metadata={}),
    )

    turn = boundary_factory._tool_loop_turn(SimpleNamespace(llm=model.engine, plugins=None), req, {})

    assert model.sent[0]["messages"][0]["content"] == SYSTEM_PROMPT
    # Code turns have their own handling after the loop, so nothing is handed on.
    assert turn == boundary_factory._LoopTurn()


@pytest.mark.parametrize(
    "question, reply",
    [
        ("thanks, that fixed it", "Good."),
        ("you're the best", "I'll take it."),
        ("is sqlite fast enough for a notes app?", "Yes."),
        ("should I rewrite the memory layer in rust?", "No."),
    ],
)
def test_a_short_complete_answer_is_not_sent_back(model, question, reply):
    """The persona asks for exactly these: the verdict first, stop at the last
    useful word. Anything under four words used to be sent back for a "take" in
    one to three sentences, which is how "Good." turned into a paragraph."""
    model.replies.append(reply)

    result = _ask(model.engine, question)

    assert len(model.sent) == 1
    assert result.response_text == reply


def test_a_reply_with_nothing_in_it_still_gets_a_second_pass(model):
    model.replies.extend(["...", REPLY])

    result = _ask(model.engine)

    assert len(model.sent) == 2
    assert result.response_text == REPLY


def test_i_dont_know_stands(model):
    """The persona calls "I don't know" a complete answer. Sending it back with
    "do not hedge, give your actual take" asks the model to guess instead, which
    is how an honest gap becomes a confident number."""
    reply = "I don't know. I can read what this machine is doing right now, if that is close enough."
    model.replies.append(reply)

    result = _ask(model.engine, "how much RAM does Ollama hold when it's idle?")

    assert len(model.sent) == 1
    assert result.response_text == reply


EXPLANATION = " ".join(f"Step {n} hands its result to the next one." for n in range(1, 8))


def test_an_explanation_he_asked_for_is_not_cut(model):
    """The cap is for rambling. Asked to walk through something, five sentences in
    it dropped the last steps, which is usually where the point is."""
    model.replies.append(EXPLANATION)

    result = _ask(model.engine, "walk me through how a turn gets answered")

    assert result.response_text == EXPLANATION


def test_an_ordinary_answer_is_still_capped(model):
    model.replies.append(EXPLANATION)

    result = _ask(model.engine, QUESTION)

    assert result.response_text.count("Step") == 5
