"""What Alice says when something goes wrong has to mean something to a person.

Degraded turns are the ones users actually read, and they were the ones written
in Alice's own vocabulary: "Falling back to language model response.",
"I misunderstood that response path.", "Working on it." — an internal route
name, an internal concept, and a promise nothing keeps.
"""

import re

import pytest

from ai.runtime.fallback_policy import FallbackGraph

# Words that name Alice's internals. A user did not ask about a route, a lane,
# or a pipeline; being told about one tells them nothing they can act on.
INTERNAL_VOCABULARY = (
    "language model",
    "llm",
    "response path",
    "fallback",
    "fast lane",
    "pipeline",
    "contract",
    "verifier",
    "decision band",
    "intent classifier",
    "tool_result",
    "plugin manager",
    "boundary",
    "orchestrator",
)

# Phrases that assert work is under way. Alice answers synchronously, so nothing
# arrives after one of these — they read as a hang.
EMPTY_PROMISES = (
    "working on it",
    "let me get back to you",
    "i'll get back to you",
    "one moment",
    "please hold",
)


def _all_fallback_messages():
    graph = FallbackGraph()
    seen = []
    for (intent, error), steps in graph._GRAPH.items():
        for step in steps:
            seen.append((f"{intent}/{error}", step.message))
    return seen


@pytest.mark.parametrize("key,message", _all_fallback_messages())
def test_recovery_messages_avoid_internal_vocabulary(key, message):
    lowered = message.lower()
    leaked = [term for term in INTERNAL_VOCABULARY if term in lowered]
    assert not leaked, f"{key} tells the user about {leaked}: {message!r}"


@pytest.mark.parametrize("key,message", _all_fallback_messages())
def test_recovery_messages_do_not_promise_work_that_never_arrives(key, message):
    lowered = message.lower()
    promised = [term for term in EMPTY_PROMISES if term in lowered]
    assert not promised, f"{key} promises {promised} but nothing follows: {message!r}"


@pytest.mark.parametrize("key,message", _all_fallback_messages())
def test_recovery_messages_are_a_real_sentence(key, message):
    assert message.strip(), f"{key} has an empty message"
    assert message.strip()[0].isupper(), f"{key} does not start a sentence: {message!r}"
    assert re.search(r"[.!?]$", message.strip()), f"{key} does not end a sentence: {message!r}"


def test_the_generic_tool_failure_message_says_what_happens_next():
    """This is the most-seen recovery message — the one every unmatched tool
    failure lands on."""
    message = FallbackGraph().first_user_message("anything", "tool_failed")
    assert message
    lowered = message.lower()
    assert "language model" not in lowered
    assert "fallback" not in lowered
