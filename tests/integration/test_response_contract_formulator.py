"""Regression tests for strict thought-vs-output response contract."""

from ai.core.response_formulator import (
    ResponseFormulator,
    ReasoningOutput,
    UserResponse,
)
from app.main import ALICE


def test_response_formulator_generate_returns_user_response_contract():
    formulator = ResponseFormulator()
    reasoning = ReasoningOutput(
        internal_summary="intent=weather:forecast | confidence=0.91",
        intent="weather:forecast",
        plan=["answer"],
        confidence=0.91,
    )

    final_response = formulator.generate(
        intent="weather:forecast",
        context={"user_input": "will it snow this week?", "response": "There is a chance later this week."},
        tool_results={"plugin": "WeatherPlugin", "action": "forecast", "response": "There is a chance later this week."},
        reasoning_output=reasoning,
        mode="final_answer_only",
    )

    assert isinstance(final_response, UserResponse)
    assert "analysis:" not in final_response.message.lower()
    assert "context:" not in final_response.message.lower()


def test_response_formulator_leak_filter_regenerates_internal_text():
    formulator = ResponseFormulator()
    reasoning = ReasoningOutput(
        internal_summary="analysis: the user wants weather details | intent=weather:forecast",
        intent="weather:forecast",
        plan=["query forecast", "answer"],
        confidence=0.88,
    )

    final_response = formulator.generate(
        intent="weather:forecast",
        context={"user_input": "is it gonna snow by any chance?", "response": "analysis: the user wants snow check"},
        tool_results={"plugin": "WeatherPlugin", "action": "forecast", "response": "analysis: key points: check snow"},
        reasoning_output=reasoning,
        mode="final_answer_only",
    )

    text = final_response.message.lower()
    assert "analysis:" not in text
    assert "the user wants" not in text
    assert "key points" not in text


def test_response_formulator_blocks_equals_delimited_internal_summary_leak():
    formulator = ResponseFormulator()
    reasoning = ReasoningOutput(
        internal_summary=(
            "intent=greeting | confidence=0.00 | route=unknown | "
            "user_input=hey alice | raw_response=hi there"
        ),
        intent="conversation:greeting",
        plan=["respond naturally"],
        confidence=0.0,
    )

    final_response = formulator.generate(
        intent="conversation:greeting",
        context={"user_input": "hey alice", "response": ""},
        tool_results={},
        reasoning_output=reasoning,
        mode="final_answer_only",
    )

    low = final_response.message.lower()
    assert "intent=" not in low
    assert "confidence=" not in low
    assert "route=" not in low
    assert "user_input=" not in low
    assert "raw_response=" not in low
    assert len(final_response.message.strip()) > 0


def test_response_formulator_project_fallback_is_structured_multiline():
    formulator = ResponseFormulator()
    reasoning = ReasoningOutput(
        internal_summary="analysis: project intent with leaked scaffold",
        intent="learning:project_ideation",
        plan=["respond"],
        confidence=0.84,
    )

    final_response = formulator.generate(
        intent="learning:project_ideation",
        context={"user_input": "i want to create an ai agent", "response": "intent=learning:project_ideation"},
        tool_results={},
        reasoning_output=reasoning,
        mode="final_answer_only",
    )

    text = final_response.message
    assert "\n\n" in text
    assert "Project Concept:" in text
    assert "Action Plan:" in text
    assert "1. Pick one domain and one measurable outcome." in text


def test_response_formulator_preserves_newlines_from_safe_candidate():
    formulator = ResponseFormulator()
    reasoning = ReasoningOutput(
        internal_summary="intent=conversation:help",
        intent="conversation:help",
        plan=["answer"],
        confidence=0.8,
    )

    final_response = formulator.generate(
        intent="conversation:help",
        context={"user_input": "help", "response": "First line.\n\nSecond line."},
        tool_results={},
        reasoning_output=reasoning,
        mode="final_answer_only",
    )

    assert final_response.message == "First line.\n\nSecond line."


def test_process_input_wrapper_blocks_internal_output_leakage():
    alice = ALICE.__new__(ALICE)
    alice.response_formulator = ResponseFormulator()
    alice._internal_reasoning_state = {}
    alice._last_routed_intent = "conversation:general"
    alice.last_intent = "conversation:general"
    alice._last_intent_confidence = 0.6
    alice._last_policy = None
    alice._last_plugin_result = {}
    alice.last_assistant_response = ""

    # Internal pipeline leaks meta text; wrapper must sanitize before returning.
    alice._process_input_internal = lambda _user_input, use_voice=False: "analysis: the user wants jarvis architecture"

    user_text = ALICE.process_input(alice, "I want to build a Jarvis-like AI")

    lowered = user_text.lower()
    assert "analysis:" not in lowered
    assert "the user wants" not in lowered
    assert len(user_text.strip()) > 0
