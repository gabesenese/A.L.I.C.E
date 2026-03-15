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
