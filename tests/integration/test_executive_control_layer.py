from ai.core.executive_controller import ExecutiveController
import pytest


def test_learning_state_builds_structured_plan() -> None:
    controller = ExecutiveController()

    state = controller.build_state(
        user_input="what is polymorphism",
        intent="learning:study_topic",
        confidence=0.88,
        entities={"topic": "polymorphism"},
        conversation_state={"conversation_goal": "learning", "depth_level": 1},
    )

    assert state.user_intent == "learning:study_topic"
    assert state.topic == "polymorphism"
    assert state.confidence >= 0.8
    assert "explain" in state.plan[0]


def test_executive_prefers_plugin_for_action_cue() -> None:
    controller = ExecutiveController()
    state = controller.build_state(
        user_input="delete note groceries",
        intent="notes:delete",
        confidence=0.74,
        entities={},
        conversation_state={},
    )

    decision = controller.decide(
        state,
        is_pure_conversation=False,
        has_explicit_action_cue=True,
        has_active_goal=False,
        force_plugins_for_notes=False,
    )

    assert decision.action == "use_plugin"
    assert decision.store_memory is True


def test_executive_requests_clarification_when_ambiguous() -> None:
    controller = ExecutiveController()
    state = controller.build_state(
        user_input="that one",
        intent="conversation:general",
        confidence=0.20,
        entities={},
        conversation_state={},
    )

    decision = controller.decide(
        state,
        is_pure_conversation=False,
        has_explicit_action_cue=False,
        has_active_goal=False,
        force_plugins_for_notes=False,
    )

    assert decision.action in ("ask_clarification", "defer")
    assert decision.store_memory is False
    if decision.action == "ask_clarification":
        assert decision.clarification_question


def test_reasoning_state_prompt_is_structured_not_cot() -> None:
    controller = ExecutiveController()
    state = controller.build_state(
        user_input="can you show an example",
        intent="conversation:question",
        confidence=0.82,
        entities={},
        conversation_state={
            "conversation_topic": "polymorphism",
            "conversation_goal": "learning",
            "user_goal": "understand polymorphism",
            "depth_level": 3,
        },
    )

    monologue = controller.format_reasoning_state(state)
    assert "Internal reasoning state" in monologue
    assert "topic: polymorphism" in monologue
    assert "plan:" in monologue


def test_decision_scoring_prefers_tools_for_explicit_actions() -> None:
    controller = ExecutiveController()
    state = controller.build_state(
        user_input="delete note groceries",
        intent="notes:delete",
        confidence=0.81,
        entities={},
        conversation_state={},
    )

    scores = controller.score_decisions(
        state,
        is_pure_conversation=False,
        has_explicit_action_cue=True,
        has_active_goal=False,
        force_plugins_for_notes=False,
    )

    assert scores["tools"] > scores["llm"]
    assert scores["tools"] > scores["clarify"]


def test_response_acceptance_gate_rejects_uncertain_generic_output() -> None:
    controller = ExecutiveController()

    result = controller.evaluate_response(
        user_input="What is polymorphism?",
        intent="learning:study_topic",
        response="Maybe it depends. I am not sure.",
        route="llm",
        context={},
    )

    assert result["accepted"] is False
    assert result["fallback_action"] in ("clarify", "safe_reply")


def test_response_acceptance_gate_accepts_relevant_answer() -> None:
    controller = ExecutiveController()

    result = controller.evaluate_response(
        user_input="What is polymorphism in OOP?",
        intent="learning:study_topic",
        response="Polymorphism in OOP means one interface can represent multiple concrete behaviors.",
        route="llm",
        context={},
    )

    assert result["accepted"] is True
    assert result["score"] >= 0.5


def test_learning_authority_can_reject_or_store() -> None:
    controller = ExecutiveController()

    reject_decision = controller.decide_learning(
        relevance=0.35,
        confidence=0.30,
        novelty=0.20,
        risk=0.80,
    )
    store_decision = controller.decide_learning(
        relevance=0.85,
        confidence=0.82,
        novelty=0.60,
        risk=0.20,
    )

    assert reject_decision == "reject"
    assert store_decision == "store"


def test_planner_authority_for_learning_turns() -> None:
    controller = ExecutiveController()
    state = controller.build_state(
        user_input="what is polymorphism",
        intent="learning:study_topic",
        confidence=0.72,
        entities={"topic": "polymorphism"},
        conversation_state={"depth_level": 2},
    )
    scores = controller.score_decisions(
        state,
        is_pure_conversation=False,
        has_explicit_action_cue=False,
        has_active_goal=False,
        force_plugins_for_notes=False,
    )

    assert controller.should_use_planner(state, scores) is True


def test_uncertainty_behavior_can_defer() -> None:
    controller = ExecutiveController()
    state = controller.build_state(
        user_input="that one",
        intent="conversation:general",
        confidence=0.30,
        entities={},
        conversation_state={},
    )
    scores = controller.score_decisions(
        state,
        is_pure_conversation=False,
        has_explicit_action_cue=False,
        has_active_goal=False,
        force_plugins_for_notes=False,
    )

    outcome = controller.uncertainty_behavior(state, scores)
    assert outcome in ("defer", "clarify", "reject")


def test_tool_veto_blocks_low_plausibility_route() -> None:
    controller = ExecutiveController()
    veto = controller.should_veto_tool_execution(
        user_input="let us brainstorm options",
        intent="weather:current",
        confidence=0.71,
        intent_plausibility=0.31,
        intent_candidates=[
            {"intent": "weather:current", "score": 0.63},
            {"intent": "conversation:general", "score": 0.58},
        ],
    )

    assert veto["veto"] is True
    assert "question" in veto


def test_tool_veto_allows_high_plausibility_action_route() -> None:
    controller = ExecutiveController()
    veto = controller.should_veto_tool_execution(
        user_input="delete my groceries note",
        intent="notes:delete",
        confidence=0.84,
        intent_plausibility=0.88,
        intent_candidates=[
            {"intent": "notes:delete", "score": 0.89},
            {"intent": "notes:list", "score": 0.36},
        ],
    )

    assert veto["veto"] is False


def test_pre_route_guard_blocks_ambiguous_low_plausibility_before_routing() -> None:
    controller = ExecutiveController()
    state = controller.build_state(
        user_input="maybe check that thing",
        intent="weather:current",
        confidence=0.41,
        entities={"_intent_plausibility": 0.33},
        conversation_state={},
    )

    guard = controller.should_preempt_for_plausibility(
        state,
        has_explicit_action_cue=False,
        intent_candidates=[
            {"intent": "weather:current", "score": 0.52},
            {"intent": "conversation:general", "score": 0.49},
        ],
    )

    assert guard["block"] is True
    assert "question" in guard


def test_runtime_controls_reduce_tool_usage_when_clarify_first() -> None:
    controller = ExecutiveController()
    state = controller.build_state(
        user_input="let's discuss options",
        intent="notes:create",
        confidence=0.48,
        entities={"_intent_plausibility": 0.40},
        conversation_state={
            "route_bias": "clarify_first",
            "tool_budget": 0,
            "planner_depth": 3,
            "planner_hint": "increase_structure_depth",
        },
    )

    scores = controller.score_decisions(
        state,
        is_pure_conversation=False,
        has_explicit_action_cue=False,
        has_active_goal=False,
        force_plugins_for_notes=False,
    )
    controls = controller.derive_runtime_controls(state, scores)

    assert controls["allow_tools"] is False
    assert controls["routing_preference"] == "clarify_first"
    assert int(controls["thinking_depth"]) >= 3


    def test_help_intent_prefers_native_scaffold_over_llm() -> None:
        controller = ExecutiveController()
        state = controller.build_state(
            user_input="i need help",
            intent="conversation:help",
            confidence=0.95,
            entities={},
            conversation_state={},
        )

        decision = controller.decide(
            state,
            is_pure_conversation=True,
            has_explicit_action_cue=False,
            has_active_goal=False,
            force_plugins_for_notes=False,
        )

        assert decision.action == "answer_direct"
        assert decision.reason == "native_conversation_scaffold"


    def test_help_opener_reduces_llm_score() -> None:
        controller = ExecutiveController()
        state = controller.build_state(
            user_input="can you help with this",
            intent="conversation:general",
            confidence=0.97,
            entities={},
            conversation_state={},
        )

        scores = controller.score_decisions(
            state,
            is_pure_conversation=True,
            has_explicit_action_cue=False,
            has_active_goal=False,
            force_plugins_for_notes=False,
        )

        assert scores["llm"] <= 0.45


def test_clarification_needed_intent_answers_instead_of_looping() -> None:
    controller = ExecutiveController()
    state = controller.build_state(
        user_input="i dont need classes, just what the file does",
        intent="conversation:clarification_needed",
        confidence=0.62,
        entities={"_intent_plausibility": 0.70},
        conversation_state={},
    )

    decision = controller.decide(
        state,
        is_pure_conversation=True,
        has_explicit_action_cue=False,
        has_active_goal=False,
        force_plugins_for_notes=False,
    )

    assert decision.action == "answer_direct"
    assert decision.reason in {
        "native_conversation_scaffold",
        "simple_conversational_native_path",
    }


@pytest.mark.parametrize(
    "utterance,intent",
    [
        ("how are you?", "conversation:general"),
        ("hello", "conversation:general"),
        ("can you help me?", "conversation:help"),
        ("i need some help", "conversation:help"),
        ("thanks", "conversation:general"),
    ],
)
def test_simple_conversational_prompts_force_native_direct_path(utterance: str, intent: str) -> None:
    controller = ExecutiveController()
    state = controller.build_state(
        user_input=utterance,
        intent=intent,
        confidence=0.95,
        entities={},
        conversation_state={},
    )

    decision = controller.decide(
        state,
        is_pure_conversation=True,
        has_explicit_action_cue=False,
        has_active_goal=False,
        force_plugins_for_notes=False,
    )

    assert decision.action == "answer_direct"
    assert decision.reason == "simple_conversational_native_path"
