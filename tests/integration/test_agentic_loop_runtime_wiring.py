from ai.core.agentic_loop import AgenticLoop
from app.main import ALICE


class _ContextStub:
    def __init__(self):
        self.status = {}

    def update_system_status(self, key, value):
        self.status[str(key)] = dict(value or {})


def test_run_agentic_control_cycle_updates_turn_state_and_status():
    alice = ALICE.__new__(ALICE)
    alice.agentic_loop = AgenticLoop(
        perceive_fn=alice._agentic_loop_perceive,
        reason_fn=alice._agentic_loop_reason,
        goal_fn=alice._agentic_loop_goal,
        decide_fn=alice._agentic_loop_decide,
        execute_fn=alice._agentic_loop_execute,
        learn_fn=alice._agentic_loop_learn,
    )
    alice.context = _ContextStub()
    alice._internal_reasoning_state = {}
    alice._last_agentic_cycle_report = {}

    report = ALICE._run_agentic_control_cycle(
        alice,
        user_input="build me an agentic planner",
        intent="conversation:help",
        entities={"topic": "agentic planner"},
        response="I can map the first implementation steps now.",
        route="llm",
        success=True,
        confidence=0.82,
        plugin_result=None,
        goal="implement agentic loop",
    )

    assert isinstance(report, dict)
    assert "cycle" in report
    assert "memory" in report
    assert report["memory"]["cycles"] == 1
    assert report["cycle"]["decision"]["action"] in {
        "respond",
        "verify_tool_outcome",
        "recover",
    }

    assert "agentic_loop" in alice._internal_reasoning_state
    assert "agentic_loop" in alice.context.status
    assert alice.context.status["agentic_loop"]["cycles"] == 1


def test_run_agentic_control_cycle_noop_when_loop_not_initialized():
    alice = ALICE.__new__(ALICE)
    alice.agentic_loop = None
    alice.context = _ContextStub()
    alice._internal_reasoning_state = {}

    report = ALICE._run_agentic_control_cycle(
        alice,
        user_input="status",
        intent="conversation:general",
        entities={},
        response="Done.",
        route="llm",
        success=True,
        confidence=0.5,
        plugin_result=None,
        goal="",
    )

    assert report == {}
    assert "agentic_loop" not in alice._internal_reasoning_state


def test_agentic_primary_authority_prefers_tool_for_action_cue():
    alice = ALICE.__new__(ALICE)
    alice.agentic_loop = AgenticLoop(
        perceive_fn=alice._agentic_loop_perceive,
        reason_fn=alice._agentic_loop_reason,
        goal_fn=alice._agentic_loop_goal,
        decide_fn=alice._agentic_loop_decide,
        execute_fn=alice._agentic_loop_execute,
        learn_fn=alice._agentic_loop_learn,
    )
    alice.context = _ContextStub()
    alice._internal_reasoning_state = {}
    alice._should_answer_first_without_clarification = lambda *_args, **_kwargs: False

    decision = ALICE._agentic_primary_authority_decision(
        alice,
        user_input="delete that note",
        intent="notes:delete",
        entities={},
        intent_confidence=0.88,
        has_action_cue=True,
        has_active_goal=False,
        execution_mode="operator",
        force_plugins_for_notes=False,
        pending_action=None,
    )

    assert decision["action"] == "use_plugin"
    assert decision["route"] == "tool"


def test_agentic_primary_authority_prefers_llm_for_conversational_lane():
    alice = ALICE.__new__(ALICE)
    alice.agentic_loop = AgenticLoop(
        perceive_fn=alice._agentic_loop_perceive,
        reason_fn=alice._agentic_loop_reason,
        goal_fn=alice._agentic_loop_goal,
        decide_fn=alice._agentic_loop_decide,
        execute_fn=alice._agentic_loop_execute,
        learn_fn=alice._agentic_loop_learn,
    )
    alice.context = _ContextStub()
    alice._internal_reasoning_state = {}
    alice._should_answer_first_without_clarification = lambda *_args, **_kwargs: True

    decision = ALICE._agentic_primary_authority_decision(
        alice,
        user_input="explain retrieval augmentation",
        intent="conversation:question",
        entities={},
        intent_confidence=0.72,
        has_action_cue=False,
        has_active_goal=False,
        execution_mode="conversational_intelligence",
        force_plugins_for_notes=False,
        pending_action=None,
    )

    assert decision["action"] == "use_llm"
    assert decision["route"] == "llm"


def test_agentic_primary_authority_clarifies_low_specificity_only_when_needed():
    alice = ALICE.__new__(ALICE)
    alice.agentic_loop = AgenticLoop(
        perceive_fn=alice._agentic_loop_perceive,
        reason_fn=alice._agentic_loop_reason,
        goal_fn=alice._agentic_loop_goal,
        decide_fn=alice._agentic_loop_decide,
        execute_fn=alice._agentic_loop_execute,
        learn_fn=alice._agentic_loop_learn,
    )
    alice.context = _ContextStub()
    alice._internal_reasoning_state = {}
    alice._should_answer_first_without_clarification = lambda *_args, **_kwargs: False

    decision = ALICE._agentic_primary_authority_decision(
        alice,
        user_input="help",
        intent="conversation:clarification_needed",
        entities={},
        intent_confidence=0.2,
        has_action_cue=False,
        has_active_goal=False,
        execution_mode="operator",
        force_plugins_for_notes=False,
        pending_action=None,
    )

    assert decision["action"] == "ask_clarification"
    assert decision["route"] == "clarify"
