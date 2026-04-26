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


def test_turn_contract_conversation_shape_is_canonical() -> None:
    controller = ExecutiveController()
    state = controller.build_state(
        user_input="how are you?",
        intent="status_inquiry",
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

    contract = controller.build_turn_contract(
        state=state,
        decision=decision,
        should_try_plugins=False,
        has_explicit_action_cue=False,
        has_active_goal=False,
    )
    payload = contract.as_dict()

    assert payload["task_type"] == "conversation"
    assert payload["chosen_route"] == "llm"
    assert payload["next_action_type"] == "respond"
    assert payload["next_action_owner"] == "response_layer"
    assert set(payload.keys()) == {
        "task_type",
        "goal",
        "constraints",
        "chosen_route",
        "success_criteria",
        "next_action",
        "next_action_type",
        "continuation_payload",
        "retry_target",
        "blocking_reason",
        "next_action_owner",
        "continuation",
    }


def test_turn_contract_direct_tool_action_shape() -> None:
    controller = ExecutiveController()
    state = controller.build_state(
        user_input="delete note groceries",
        intent="notes:delete",
        confidence=0.88,
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

    contract = controller.build_turn_contract(
        state=state,
        decision=decision,
        should_try_plugins=True,
        has_explicit_action_cue=True,
        has_active_goal=False,
    )

    assert contract.task_type == "direct tool action"
    assert contract.chosen_route == "tool"
    assert any("tool" in x.lower() for x in contract.success_criteria)


def test_turn_contract_clarification_required_shape() -> None:
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

    contract = controller.build_turn_contract(
        state=state,
        decision=decision,
        should_try_plugins=False,
        has_explicit_action_cue=False,
        has_active_goal=False,
    )

    assert contract.task_type in {"clarification-required", "blocked/escalated"}
    if contract.task_type == "clarification-required":
        assert contract.chosen_route == "clarify"


def test_turn_state_machine_produces_tool_route_for_plugin_decision() -> None:
    controller = ExecutiveController()
    state = controller.build_state(
        user_input="delete note groceries",
        intent="notes:delete",
        confidence=0.86,
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

    sm = controller.run_turn_state_machine(
        state=state,
        decision=decision,
        has_explicit_action_cue=True,
        has_active_goal=False,
        pre_route_blocked=False,
        tool_vetoed=False,
    )

    assert sm.chosen_route == "tool"
    assert sm.should_try_plugins is True
    assert sm.terminal_action == "proceed"


def test_turn_state_machine_returns_clarify_terminal_for_pre_route_block() -> None:
    controller = ExecutiveController()
    state = controller.build_state(
        user_input="that one",
        intent="conversation:general",
        confidence=0.12,
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

    sm = controller.run_turn_state_machine(
        state=state,
        decision=decision,
        has_explicit_action_cue=False,
        has_active_goal=False,
        pre_route_blocked=True,
        tool_vetoed=False,
    )

    assert sm.chosen_route == "clarify"
    assert sm.should_try_plugins is False
    assert sm.terminal_action == "clarify"


def test_post_execution_state_machine_marks_retry_for_retryable_failure() -> None:
    controller = ExecutiveController()
    state = controller.build_state(
        user_input="delete note groceries",
        intent="notes:delete",
        confidence=0.82,
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
    pre = controller.run_turn_state_machine(
        state=state,
        decision=decision,
        has_explicit_action_cue=True,
        has_active_goal=False,
        pre_route_blocked=False,
        tool_vetoed=False,
    )
    outcome = controller.build_execution_outcome(
        contract=pre.contract,
        tool_success=False,
        goal_advanced=False,
        verification_passed=False,
        recommended_next_action="retry",
        retryable=True,
        issues=["tool_timeout"],
        metadata={"plugin": "notes"},
    )

    post = controller.run_post_execution_state_machine(
        pre_execution=pre,
        outcome=outcome,
    )

    assert post.phase == "retry"
    assert post.should_retry is True
    assert post.contract.as_dict()["next_action_type"] == "retry_tool"


def test_post_execution_state_machine_marks_completed_after_verified_goal_advance() -> None:
    controller = ExecutiveController()
    state = controller.build_state(
        user_input="delete note groceries",
        intent="notes:delete",
        confidence=0.90,
        entities={},
        conversation_state={"conversation_goal": "organize notes"},
    )
    decision = controller.decide(
        state,
        is_pure_conversation=False,
        has_explicit_action_cue=True,
        has_active_goal=True,
        force_plugins_for_notes=False,
    )
    pre = controller.run_turn_state_machine(
        state=state,
        decision=decision,
        has_explicit_action_cue=True,
        has_active_goal=True,
        pre_route_blocked=False,
        tool_vetoed=False,
    )
    outcome = controller.build_execution_outcome(
        contract=pre.contract,
        tool_success=True,
        goal_advanced=True,
        verification_passed=True,
        recommended_next_action="continue",
        retryable=False,
        issues=[],
        metadata={"plugin": "notes"},
    )

    post = controller.run_post_execution_state_machine(
        pre_execution=pre,
        outcome=outcome,
    )

    assert post.phase == "completed"
    assert post.should_retry is False
    assert post.should_replan is False
    assert post.contract.as_dict()["next_action_type"] == "continue_goal"


def test_post_execution_state_machine_escalates_unverified_non_retryable_tool_turn() -> None:
    controller = ExecutiveController()
    state = controller.build_state(
        user_input="delete note groceries",
        intent="notes:delete",
        confidence=0.88,
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
    pre = controller.run_turn_state_machine(
        state=state,
        decision=decision,
        has_explicit_action_cue=True,
        has_active_goal=False,
        pre_route_blocked=False,
        tool_vetoed=False,
    )
    outcome = controller.build_execution_outcome(
        contract=pre.contract,
        tool_success=False,
        goal_advanced=False,
        verification_passed=False,
        recommended_next_action="",
        retryable=False,
        issues=["verification_failed"],
        metadata={"plugin": "notes"},
    )

    post = controller.run_post_execution_state_machine(
        pre_execution=pre,
        outcome=outcome,
    )

    assert post.phase == "escalated"
    assert post.contract.as_dict()["next_action_type"] == "escalate"


def test_executive_keeps_greeting_on_native_conversational_path() -> None:
    controller = ExecutiveController()
    state = controller.build_state(
        user_input="hi alice",
        intent="greeting",
        confidence=0.90,
        entities={},
        conversation_state={
            "active_goal_stack": [
                {
                    "goal_id": "goal::stale",
                    "title": "old unfinished task",
                    "status": "active",
                }
            ]
        },
    )

    decision = controller.decide(
        state,
        is_pure_conversation=True,
        has_explicit_action_cue=False,
        has_active_goal=True,
        force_plugins_for_notes=False,
    )

    assert decision.action == "answer_direct"
    assert decision.reason == "greeting_native_priority"


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
    assert result["fallback_action"] in ("clarify", "safe_reply", "revise_answer")


def test_response_acceptance_gate_prefers_refine_for_open_learning_goal() -> None:
    controller = ExecutiveController()

    result = controller.evaluate_response(
        user_input="i want to learn more about agentic ai",
        intent="conversation:goal_statement",
        response="In general, maybe it depends, but agentic systems plan and act toward goals.",
        route="llm",
        context={},
    )

    assert result["accepted"] is False
    assert result["fallback_action"] == "revise_answer"
    assert result.get("fallback_reason") in {
        "low_confidence_answer_needs_refinement",
        "low_confidence_answer_requiring_retry",
    }


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


def test_planner_authority_for_beginner_explanation_help_turn() -> None:
    controller = ExecutiveController()
    state = controller.build_state(
        user_input="i am beginner and explain this step by step",
        intent="conversation:help",
        confidence=0.70,
        entities={},
        conversation_state={"depth_level": 1},
    )
    scores = controller.score_decisions(
        state,
        is_pure_conversation=True,
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
    low = str(guard["question"]).lower()
    assert "tool" not in low
    assert "conversational" not in low


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


def test_help_intent_no_longer_forces_native_scaffold() -> None:
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

    assert decision.action == "use_llm"
    assert decision.reason == "score_llm"


def test_help_opener_no_longer_reduces_llm_score() -> None:
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

    assert scores["llm"] >= 0.75


def test_status_inquiry_forces_simple_native_path() -> None:
    controller = ExecutiveController()
    state = controller.build_state(
        user_input="how are you?",
        intent="status_inquiry",
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


def test_help_intent_with_educational_question_does_not_force_simple_native_path() -> None:
    controller = ExecutiveController()
    state = controller.build_state(
        user_input="give me a brief summary of how ai works",
        intent="conversation:help",
        confidence=0.93,
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

    assert decision.reason != "simple_conversational_native_path"


def test_beginner_explanation_help_request_routes_to_reasoning() -> None:
    controller = ExecutiveController()
    state = controller.build_state(
        user_input="i am beginner so i want an explanation",
        intent="conversation:help",
        confidence=0.70,
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

    assert decision.action == "use_llm"
    assert decision.reason == "score_llm"


def test_pending_followup_slot_short_answer_avoids_scaffold_loop() -> None:
    controller = ExecutiveController()
    state = controller.build_state(
        user_input="NLP",
        intent="conversation:general",
        confidence=0.72,
        entities={},
        conversation_state={
            "pending_followup_slot": True,
            "pending_followup_slot_name": "project_subdomain",
        },
    )

    decision = controller.decide(
        state,
        is_pure_conversation=True,
        has_explicit_action_cue=False,
        has_active_goal=False,
        force_plugins_for_notes=False,
    )

    assert decision.action == "use_llm"
    assert decision.reason == "pending_followup_slot_answer"


def test_pending_followup_slot_ordinal_answer_avoids_scaffold_loop() -> None:
    controller = ExecutiveController()
    state = controller.build_state(
        user_input="the second one",
        intent="conversation:general",
        confidence=0.66,
        entities={},
        conversation_state={
            "pending_followup_slot": True,
            "pending_followup_slot_name": "project_subdomain",
            "pending_followup_slot_state": {
                "expected_answer_shape": "ordinal_or_short_phrase",
                "slot": "project_subdomain",
            },
        },
    )

    decision = controller.decide(
        state,
        is_pure_conversation=True,
        has_explicit_action_cue=False,
        has_active_goal=False,
        force_plugins_for_notes=False,
    )

    assert decision.action == "use_llm"
    assert decision.reason == "pending_followup_slot_answer"


def test_clear_informational_request_prefers_direct_answer() -> None:
    controller = ExecutiveController()
    state = controller.build_state(
        user_input="explain nlp intent routing in simple terms for beginners",
        intent="conversation:help",
        confidence=0.88,
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
    assert decision.reason == "clear_informational_request"


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

    assert decision.action == "use_llm"
    assert decision.reason == "clarification_answer_requested"


def test_build_state_threads_goal_blockers_and_next_action_from_stack() -> None:
    controller = ExecutiveController()
    state = controller.build_state(
        user_input="continue",
        intent="conversation:general",
        confidence=0.74,
        entities={},
        conversation_state={
            "active_goal_stack": [
                {
                    "goal_id": "g1",
                    "title": "Ship parser",
                    "status": "blocked",
                    "next_action": "ask for fixture format",
                    "blockers": ["fixture format missing"],
                }
            ],
            "current_goal": {
                "goal_id": "g1",
                "title": "Ship parser",
                "status": "blocked",
                "next_action": "ask for fixture format",
                "blockers": ["fixture format missing"],
            },
        },
    )

    assert state.goal_status == "blocked"
    assert state.goal_next_action == "ask for fixture format"
    assert "fixture format missing" in state.goal_blockers
    assert state.goal_active_count >= 1


def test_goal_blockers_and_next_action_shape_routing_scores() -> None:
    controller = ExecutiveController()
    actionable_state = controller.build_state(
        user_input="continue",
        intent="conversation:general",
        confidence=0.72,
        entities={"_intent_plausibility": 0.80},
        conversation_state={
            "active_goal_stack": [
                {
                    "goal_id": "g-action",
                    "title": "Clean project notes",
                    "status": "active",
                    "next_action": "archive stale entries",
                    "blockers": [],
                }
            ],
            "current_goal": {
                "goal_id": "g-action",
                "title": "Clean project notes",
                "status": "active",
                "next_action": "archive stale entries",
                "blockers": [],
            },
        },
    )
    blocked_state = controller.build_state(
        user_input="continue",
        intent="conversation:general",
        confidence=0.72,
        entities={"_intent_plausibility": 0.80},
        conversation_state={
            "active_goal_stack": [
                {
                    "goal_id": "g-blocked",
                    "title": "Clean project notes",
                    "status": "blocked",
                    "next_action": "",
                    "blockers": ["waiting for project name"],
                }
            ],
            "current_goal": {
                "goal_id": "g-blocked",
                "title": "Clean project notes",
                "status": "blocked",
                "next_action": "",
                "blockers": ["waiting for project name"],
            },
        },
    )

    actionable_scores = controller.score_decisions(
        actionable_state,
        is_pure_conversation=False,
        has_explicit_action_cue=False,
        has_active_goal=True,
        force_plugins_for_notes=False,
    )
    blocked_scores = controller.score_decisions(
        blocked_state,
        is_pure_conversation=False,
        has_explicit_action_cue=False,
        has_active_goal=True,
        force_plugins_for_notes=False,
    )

    assert actionable_scores["tools"] > blocked_scores["tools"]
    assert blocked_scores["clarify"] > actionable_scores["clarify"]


@pytest.mark.parametrize(
    "utterance,intent",
    [
        ("how are you?", "conversation:general"),
        ("hello", "conversation:general"),
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


def test_pre_route_guard_allows_rich_conceptual_prompt_even_when_plausibility_is_low() -> None:
    controller = ExecutiveController()
    state = controller.build_state(
        user_input="let's imagine how assistant would be created with today's technology no fiction",
        intent="notes:list",
        confidence=0.40,
        entities={"_intent_plausibility": 0.31},
        conversation_state={},
    )

    guard = controller.should_preempt_for_plausibility(
        state,
        has_explicit_action_cue=False,
        intent_candidates=[
            {"intent": "notes:list", "score": 0.51},
            {"intent": "conversation:question", "score": 0.49},
        ],
    )

    assert guard["block"] is False
    assert guard["reason"] == "rich_conceptual_prompt"


def test_rich_conceptual_prompt_with_clarification_intent_prefers_direct_answer_mode() -> None:
    controller = ExecutiveController()
    state = controller.build_state(
        user_input="let's imagine how assistant would be created with today's technology no fiction",
        intent="conversation:clarification_needed",
        confidence=0.63,
        entities={"_intent_plausibility": 0.55},
        conversation_state={},
    )

    decision = controller.decide(
        state,
        is_pure_conversation=True,
        has_explicit_action_cue=False,
        has_active_goal=False,
        force_plugins_for_notes=False,
    )

    assert decision.action == "use_llm"
    assert decision.reason == "conceptual_build_question"


def test_rich_conceptual_build_prompt_with_clarification_bias_still_uses_fresh_reasoning() -> None:
    controller = ExecutiveController()
    state = controller.build_state(
        user_input="how can i create an ai just like assistant but with todays technology",
        intent="conversation:clarification_needed",
        confidence=0.52,
        entities={"_intent_plausibility": 0.44},
        conversation_state={
            "conversation_goal": "clarifying",
            "user_goal": "understand architecture",
        },
    )

    decision = controller.decide(
        state,
        is_pure_conversation=True,
        has_explicit_action_cue=False,
        has_active_goal=True,
        force_plugins_for_notes=False,
    )

    assert decision.action == "use_llm"
    assert decision.reason == "conceptual_build_question"


def test_short_framework_overview_prompt_routes_to_llm_without_plugin_attempt() -> None:
    controller = ExecutiveController()
    state = controller.build_state(
        user_input="agentic ai frameworks",
        intent="conversation:help",
        confidence=0.70,
        entities={"_intent_plausibility": 0.60},
        conversation_state={},
    )

    decision = controller.decide(
        state,
        is_pure_conversation=True,
        has_explicit_action_cue=False,
        has_active_goal=False,
        force_plugins_for_notes=False,
    )

    assert decision.action == "use_llm"
    assert decision.reason == "short_framework_overview"


def test_short_framework_prompt_with_explicit_action_cue_does_not_force_llm() -> None:
    controller = ExecutiveController()
    state = controller.build_state(
        user_input="search agentic ai frameworks",
        intent="conversation:help",
        confidence=0.72,
        entities={"_intent_plausibility": 0.73},
        conversation_state={},
    )

    decision = controller.decide(
        state,
        is_pure_conversation=False,
        has_explicit_action_cue=True,
        has_active_goal=False,
        force_plugins_for_notes=False,
    )

    assert decision.reason != "short_framework_overview"


def test_answerability_direct_question_routes_to_answer_direct() -> None:
    controller = ExecutiveController()
    state = controller.build_state(
        user_input="what's the difference between an ai agent and an ai assistant?",
        intent="conversation:help",
        confidence=0.52,
        entities={"_intent_plausibility": 0.61},
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
        "answerability_gate_direct_question",
        "clear_informational_request",
    }


def test_response_acceptance_gate_sets_llm_failure_reason_for_answerable_question() -> None:
    controller = ExecutiveController()

    result = controller.evaluate_response(
        user_input="what is nlp?",
        intent="conversation:question",
        response="Maybe. I am not sure.",
        route="llm",
        context={},
    )

    assert result["accepted"] is False
    assert result["fallback_action"] in {"safe_reply", "revise_answer"}
    assert result.get("fallback_reason") == "llm_failed_after_answer_directly"
