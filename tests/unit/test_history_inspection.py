"""The detector for machinery in Alice's own transcript.

scripts/quality_harness.py can only observe this against a live model, so the
detector itself is proven here instead.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def harness():
    spec = importlib.util.spec_from_file_location("quality_harness", PROJECT_ROOT / "scripts" / "quality_harness.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["quality_harness"] = module
    spec.loader.exec_module(module)
    return module


def _alice(history):
    return SimpleNamespace(llm=SimpleNamespace(conversation_history=history))


def _turn(role, content):
    return {"role": role, "content": content}


def test_a_clean_conversation_reports_nothing(harness):
    history = [
        _turn("user", "is sqlite going to be fast enough for this?"),
        _turn("assistant", "For a single user on one machine, easily."),
        _turn("user", "what would you do first?"),
        _turn("assistant", "Fix the schema drift."),
    ]
    report = harness.inspect_conversation_history(_alice(history))
    assert report["turns"] == 2
    assert report["machine_prompts_in_history"] == 0


@pytest.mark.parametrize(
    "prompt",
    [
        "Extract the goal from the following user message. Return only JSON.",
        "You are a natural language generator for Alice. Convert the payload below.",
        "Classify the intent of this utterance.",
        "Task: rewrite the response to be concise.\nContext: weather",
        "Respond with only the score as an integer.",
        "Summarize the conversation so far. Do not add commentary.",
    ],
)
def test_an_internal_prompt_is_recognised(harness, prompt):
    """These are prompts Alice wrote to herself. Replayed as conversation, the
    model imitates their register — clipped, corrective, instruction-shaped."""
    report = harness.inspect_conversation_history(_alice([_turn("user", prompt), _turn("assistant", "ok")]))
    assert report["machine_prompts_in_history"] == 1, prompt


def test_the_share_of_the_transcript_is_reported(harness):
    history = []
    for _ in range(3):
        history += [_turn("user", "what do you make of this?"), _turn("assistant", "Not much.")]
    history += [_turn("user", "Extract the goal. Return only JSON."), _turn("assistant", "{}")]

    report = harness.inspect_conversation_history(_alice(history))
    assert report["turns"] == 4
    assert report["machine_prompts_in_history"] == 1
    assert report["share"] == 0.25


def test_assistant_turns_are_not_scanned(harness):
    """Alice legitimately saying "I'll summarize that" is not machinery."""
    history = [_turn("user", "can you sum that up?"), _turn("assistant", "Summarize: three files changed.")]
    assert harness.inspect_conversation_history(_alice(history))["machine_prompts_in_history"] == 0


def test_an_engine_with_no_history_is_handled(harness):
    assert harness.inspect_conversation_history(_alice([]))["turns"] == 0
    assert harness.inspect_conversation_history(SimpleNamespace())["turns"] == 0


def test_examples_are_returned_for_the_report(harness):
    history = [_turn("user", "Extract the goal from the message below. Return only JSON."), _turn("assistant", "{}")]
    report = harness.inspect_conversation_history(_alice(history))
    assert report["examples"]
    assert "Extract the goal" in report["examples"][0]


def test_an_ordinary_question_beginning_with_a_verb_is_not_machinery(harness):
    """Guard against over-matching: a person saying "summarize this for me" is a
    person, and flagging it would make the instrument useless."""
    history = [_turn("user", "summarise what we decided, in a sentence"), _turn("assistant", "SQLite, no rewrite.")]
    report = harness.inspect_conversation_history(_alice(history))
    assert report["machine_prompts_in_history"] == 0
