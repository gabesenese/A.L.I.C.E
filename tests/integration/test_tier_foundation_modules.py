from ai.core.activity_monitor import ActivityMonitor
from ai.core.conversation_memory import ConversationMemory
from ai.core.dialogue_state_machine import DialogueState, DialogueStateMachine
from ai.core.implicit_intent_detector import ImplicitIntentDetector
from ai.core.temporal_reasoner import TemporalReasoner


def test_conversation_memory_contextual_augmentation_for_vague_followup():
    mem = ConversationMemory(max_turns=20)
    mem.add_turn(
        user_input="I am studying machine learning optimization",
        intent="learning:study_topic",
        context_extracted={"topic": "machine learning optimization"},
    )

    augmented = mem.build_contextual_input("Can you help me with that?")
    assert "Previous context topic" in augmented
    assert "machine learning optimization" in augmented


def test_implicit_intent_detector_weather_soft_request():
    detector = ImplicitIntentDetector()
    matches = detector.detect("maybe check the weather for me")
    assert matches
    assert matches[0].intent == "weather:current"
    assert matches[0].confidence >= 0.8


def test_temporal_reasoner_parses_recurring_task():
    reasoner = TemporalReasoner()
    task = reasoner.parse_temporal_task("Give me daily summaries at 9am")
    assert task is not None
    assert task.frequency in {"daily", "once"}
    assert task.when_iso


def test_dialogue_state_machine_clarifying_timeout_returns_ready():
    machine = DialogueStateMachine(max_clarifying_turns=2)
    machine.observe_intent("conversation:clarification_needed")
    state = machine.observe_intent("conversation:clarification_needed")
    assert state == DialogueState.READY


def test_activity_monitor_generates_debug_suggestion():
    monitor = ActivityMonitor()
    monitor.observe("debug")
    suggestions = monitor.proactive_suggestions()
    assert any("debug" in s.lower() or "root causes" in s.lower() for s in suggestions)
