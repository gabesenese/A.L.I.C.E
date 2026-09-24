"""Every turn the user was shown is in the transcript the next turn is built from.

Only the chat path recorded its exchange. A turn answered by a plugin, narrated
or not, never reached the history, so after "what's the weather?" the next
question went to a model with no weather in front of it, and the continuity
guard, which reads the same history, deleted a reply's callback to anything said
on such a turn: "SQLite already handles far more notes than you'll write" was cut
to "Overkill." because the turn where he said "sqlite" had gone to a plugin.
"""

import pytest

from ai.core.llm_engine import LLMConfig, LocalLLMEngine
from ai.runtime.alice_contract_factory import build_runtime_boundaries
from ai.runtime.contract_pipeline import ContractPipeline
from ai.runtime.turn_orchestrator import run_default_turn
from tests.integration.test_contract_pipeline import _FakeAlice


@pytest.fixture
def alice(monkeypatch):
    engine = LocalLLMEngine(LLMConfig(model="test-model"))
    replies = []
    monkeypatch.setattr(engine, "_ensure_service_probed", lambda: None)
    monkeypatch.setattr(engine, "_available_models", [])
    monkeypatch.setattr(
        engine,
        "_post_with_retry",
        lambda url, payload, what="": {"message": {"content": replies.pop(0) if replies else ""}},
    )
    fake = _FakeAlice()
    fake.llm = engine
    fake.plugins = None  # no tool chaining; the pipeline's own plugin stub still answers
    fake.contract_pipeline = ContractPipeline(build_runtime_boundaries(fake))
    fake.structured_logger = None
    fake.replies = replies
    return fake


def _spoken(alice):
    return [(m["role"], m["content"]) for m in alice.llm.conversation_history]


def test_a_plugin_turn_is_recorded_as_the_user_saw_it(alice):
    shown = run_default_turn(alice, "weather in boston")

    assert shown
    assert _spoken(alice) == [("user", "weather in boston"), ("assistant", shown)]


def test_a_chat_turn_is_recorded_once(alice):
    alice.replies.append("SQLite. One user, one machine, and it ships with Python.")

    shown = run_default_turn(alice, "should I use sqlite or postgres?")

    assert _spoken(alice) == [("user", "should I use sqlite or postgres?"), ("assistant", shown)]


def test_the_transcript_holds_what_was_shown_not_the_draft(alice):
    """The chat path records the model's text; the user may be shown less."""
    alice.replies.append("Great question! SQLite. One user, one machine.")

    shown = run_default_turn(alice, "should I use sqlite or postgres?")

    assert _spoken(alice)[-1] == ("assistant", shown)
    assert not shown.startswith("Great question")


def test_asking_the_same_thing_twice_is_two_turns(alice):
    alice.replies.extend(["Because the index is rebuilt.", "Because the index is rebuilt, every time."])

    run_default_turn(alice, "why is it slow?")
    run_default_turn(alice, "why is it slow?")

    assert [role for role, _ in _spoken(alice)] == ["user", "assistant", "user", "assistant"]
