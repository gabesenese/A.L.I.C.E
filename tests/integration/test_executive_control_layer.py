from ai.core.executive_controller import ExecutiveController


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

    assert decision.action == "ask_clarification"
    assert decision.store_memory is False
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
