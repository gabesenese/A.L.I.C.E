"""Conversation history should be the conversation, and nothing else.

`chat()` used to append the prompt and the reply to `conversation_history`
unconditionally — including when the caller passed `use_history=False`, which is
precisely the flag that marks a call as machinery rather than conversation.

Twelve internal callers pass it: goal extraction, plan generation, the learning
engine, the response-variance engine, greeting scaffolds, weather wrapping, the
scenario generator, the training evaluators. Every one of their prompts, and the
model's reply to it, landed in the transcript Alice replays to herself. On the
next real turn the model was shown that as "what we have been talking about" and
did the obvious thing: imitated the register. Instruction-shaped text in,
instruction-shaped text out.

These are behavioural, not structural — they drive `chat()` against a stubbed
transport rather than pattern-matching its source, so a later refactor of the
same guard does not break them.
"""

import re
from pathlib import Path

import pytest

from ai.core.llm_engine import LLMConfig, LocalLLMEngine

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def engine(monkeypatch):
    """An engine whose transport always answers, so nothing touches the network."""
    built = LocalLLMEngine(LLMConfig(model="test-model"))
    monkeypatch.setattr(built, "_ensure_service_probed", lambda: None)
    monkeypatch.setattr(built, "_available_models", [])
    monkeypatch.setattr(
        built,
        "_post_with_retry",
        lambda url, payload, what="": {"message": {"content": "a reply"}},
    )
    return built


MACHINE_PROMPT = "Extract the goal from the following message. Return only JSON."


def test_a_real_turn_is_recorded(engine):
    engine.chat("is sqlite fast enough?", use_history=True)
    assert [e["content"] for e in engine.conversation_history] == ["is sqlite fast enough?", "a reply"]


def test_a_machine_prompt_is_not_recorded(engine):
    """use_history=False is how a caller says "this is not a conversation"."""
    engine.chat(MACHINE_PROMPT, use_history=False)
    assert engine.conversation_history == []


def test_machinery_does_not_contaminate_a_real_conversation(engine):
    """The failure that mattered: machinery interleaved with real turns, so the
    transcript Alice replays is half prompts she wrote to herself."""
    engine.chat("what should I work on?", use_history=True)
    engine.chat(MACHINE_PROMPT, use_history=False)
    engine.chat("Classify the intent of the user's utterance.", use_history=False)
    engine.chat("and after that?", use_history=True)

    spoken = [e["content"] for e in engine.conversation_history if e["role"] == "user"]
    assert spoken == ["what should I work on?", "and after that?"]


def test_recording_can_be_requested_independently(engine):
    """The rare caller that wants one sense without the other."""
    engine.chat("a question", use_history=False, record_history=True)
    assert [e["content"] for e in engine.conversation_history] == ["a question", "a reply"]

    engine.conversation_history.clear()
    engine.chat("another", use_history=True, record_history=False)
    assert engine.conversation_history == []


def test_an_empty_reply_is_never_recorded(engine, monkeypatch):
    monkeypatch.setattr(engine, "_post_with_retry", lambda url, payload, what="": {"message": {"content": "  "}})
    engine.chat("a question", use_history=True)
    assert engine.conversation_history == []


def test_a_regenerated_reply_amends_rather_than_duplicates(engine):
    """The retry gate reruns a turn when the first pass hedged. Recording both
    would put the question in the transcript twice with two different answers."""
    engine.chat("what do you make of this?", use_history=True)
    engine.amend_last_reply("My actual take.")

    assert [e["content"] for e in engine.conversation_history] == [
        "what do you make of this?",
        "My actual take.",
    ]


def test_amending_an_empty_transcript_is_harmless(engine):
    assert engine.amend_last_reply("anything") is False


def test_the_conversational_context_window_fits_its_own_preamble():
    """chat() asked for num_ctx 4096 while its own system prompt, companion
    context and identity blocks run well past a thousand tokens — so history was
    squeezed out of its own context window and Alice lost the thread inside a
    single sitting. Every turn independent of the last is what a terminal is."""
    text = (PROJECT_ROOT / "ai" / "core" / "llm_engine.py").read_text(encoding="utf-8")
    start = text.index("    def chat(")
    body = text[start : text.index("\n    def ", start + 10)]

    match = re.search(r'"num_ctx"\s*:\s*(\d+)', body)
    assert match, "chat() no longer sets num_ctx explicitly"
    assert int(match.group(1)) >= 8192, f"num_ctx is {match.group(1)} on the conversational path"
