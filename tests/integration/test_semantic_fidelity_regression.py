from dataclasses import dataclass
from types import SimpleNamespace

from ai.context_resolver import ContextResolver
from ai.core.executive_controller import ExecutiveController
from app.main import ALICE


EXACT_PROMPT = (
    "i want to learn the foundations an advanced assistant system should have "
    "in today's world with no fiction"
)

EXACT_LOG_PROMPT = "let's imagine how assistant would be created with today's technology no fiction"
EXACT_TONY_PROMPT = "let's imagine how fictional inventor would have created assistant with todays technology, no fiction"
EXACT_CREATE_PROMPT = "how can i create an ai just like assistant but with todays technology"
EXACT_FICTIONAL_INVENTOR_PROMPT = EXACT_TONY_PROMPT
EXACT_FRAMEWORKS_PROMPT = "what practical frameworks should i use for agentic ai in production with no fiction"


@dataclass
class _FakeResolveResult:
    rewritten_input: str
    resolved_bindings: dict
    unresolved_pronouns: list


def test_context_resolver_keeps_raw_input_when_rewrite_has_placeholder_noise(monkeypatch):
    resolver = ContextResolver()

    def _fake_resolve(user_input, state):
        return _FakeResolveResult(
            rewritten_input="the foundations person 'an ai' assistant general_assistance",
            resolved_bindings={"it": "assistant"},
            unresolved_pronouns=[],
        )

    monkeypatch.setattr(resolver.reference_resolver, "resolve", _fake_resolve)
    monkeypatch.setattr(resolver.ambiguity_detector, "should_clarify", lambda **kwargs: False)

    result = resolver.resolve(EXACT_PROMPT, {"referenced_entities": []})

    assert result.rewritten_input == EXACT_PROMPT
    assert result.rewrite_confidence == 0.0


def test_semantic_fidelity_guard_rejects_programming_drift_response():
    controller = ExecutiveController()
    bad = (
        "Polymorphism and interface inheritance are the foundations to understand this topic."
    )

    evaluation = controller.evaluate_response(
        user_input=EXACT_PROMPT,
        intent="conversation:question",
        response=bad,
        route="llm",
        context={},
    )

    assert evaluation["accepted"] is False
    assert evaluation["reason"] in {
        "semantic_drift_programming_domain",
        "semantic_core_missing",
    }


def test_semantic_fidelity_guard_accepts_on_topic_foundations_response():
    controller = ExecutiveController()
    good = (
        "A real-world assistant system needs natural language understanding, "
        "memory, planning, execution, verification, and bounded autonomy."
    )

    evaluation = controller.evaluate_response(
        user_input=EXACT_PROMPT,
        intent="conversation:question",
        response=good,
        route="llm",
        context={},
    )

    assert evaluation["accepted"] is True


def test_study_flow_not_promoted_for_broad_conceptual_question_without_explicit_study_request():
    alice = ALICE.__new__(ALICE)

    intent, entities = alice._promote_learning_goal_intent(
        EXACT_PROMPT,
        "conversation:question",
        {},
    )

    assert intent == "conversation:question"
    assert entities == {}


def test_native_conceptual_mode_returns_direct_foundation_answer():
    alice = ALICE.__new__(ALICE)

    response = alice._native_conceptual_answer(
        EXACT_PROMPT,
        "conversation:question",
    )

    assert response is not None
    low = response.lower()
    assert "assistant" in low
    assert "memory" in low
    assert "planning" in low
    assert "execution" in low
    assert "autonomy" in low
    assert "person 'an ai'" not in low
    assert "polymorphism" not in low


def test_native_conceptual_mode_returns_nlp_algorithm_answer():
    alice = ALICE.__new__(ALICE)

    response = alice._native_conceptual_answer(
        "give me some nlp algorithms",
        "conversation:help",
    )

    assert response is not None
    low = response.lower()
    assert "nlp" in low
    assert "transformer" in low
    assert "tf-idf" in low


def test_help_opener_detection_does_not_downgrade_substantive_request():
    alice = ALICE.__new__(ALICE)

    detected = alice._is_help_opener_input(
        "help me with my ai project, i need to know some nlp algorithms",
        "conversation:help",
    )

    assert detected is False


def test_deterministic_knowledge_fallback_handles_embedding_model_request():
    alice = ALICE.__new__(ALICE)

    response = alice._deterministic_knowledge_fallback(
        "what are some embeddings models i should use",
        "conversation:question",
    )

    assert response is not None
    low = response.lower()
    assert "embedding" in low
    assert "tf-idf" in low or "transformer" in low or "retrieval" in low


def test_deterministic_knowledge_fallback_handles_nlp_learning_overview_request():
    alice = ALICE.__new__(ALICE)

    response = alice._deterministic_knowledge_fallback(
        "i want to learn abit more about nlp",
        "conversation:help",
    )

    assert response is not None
    low = response.lower()
    assert "nlp" in low
    assert "intent" in low
    assert "conversation flow" in low


def test_answerability_gate_detects_specific_domain_questions_and_ignores_ambiguous_phrasing():
    alice = ALICE.__new__(ALICE)

    assert alice._is_answerability_direct_question(
        "how does optimizer training work in nlp models?"
    ) is True
    assert alice._is_answerability_direct_question(
        "i want to know the difference between the agentic ai and generative ai"
    ) is True
    assert alice._is_answerability_direct_question(
        "how does optimizer stuff work?"
    ) is False


def test_goal_statement_promotion_demotes_direct_difference_prompt_to_question_intent():
    alice = ALICE.__new__(ALICE)

    intent, entities, confidence = alice._promote_goal_statement_intent(
        user_input="i want to know the difference between the agentic ai and generative ai",
        intent="conversation:goal_statement",
        entities={"goal": "difference between agentic ai and generative ai"},
        intent_confidence=0.84,
    )

    assert intent == "conversation:question"
    assert entities.get("goal") == "difference between agentic ai and generative ai"
    assert confidence >= 0.84


def test_answerability_gate_fallback_returns_substantive_answer_for_unknown_direct_question():
    alice = ALICE.__new__(ALICE)

    response = alice._answerability_gate_fallback_response(
        "what is agentic autonomy?"
    )

    low = response.lower()
    assert "clarify" not in low
    assert "exact result" not in low
    assert "agentic autonomy" in low


def test_answerability_gate_fallback_handles_existing_framework_inventory_prompt():
    alice = ALICE.__new__(ALICE)

    response = alice._answerability_gate_fallback_response(
        "explain to me each existing framework"
    )

    low = response.lower()
    assert "runtime contract pipeline" in low
    assert "executive decision framework" in low
    assert "bounded autonomy framework" in low
    assert "best handled with a concise explanation" not in low


def test_answerability_gate_forces_answer_first_without_clarification():
    alice = ALICE.__new__(ALICE)

    assert alice._should_answer_first_without_clarification(
        "how does optimizer training work in nlp models?",
        "conversation:question",
    ) is True


def test_fast_lane_sanitizer_removes_clarification_dead_end_for_answerable_question():
    alice = ALICE.__new__(ALICE)

    response = alice._sanitize_fast_lane_response(
        response=(
            "Please clarify what you mean. "
            "In NLP model training, the optimizer updates weights using gradient steps."
        ),
        user_input="how does optimizer training work in nlp model?",
        intent="learning:explanation_request",
    )

    low = response.lower()
    assert "please clarify" not in low
    assert "optimizer" in low


def test_fast_lane_sanitizer_repairs_abrupt_endings_in_algorithm_turns():
    alice = ALICE.__new__(ALICE)

    response = alice._sanitize_fast_lane_response(
        response="For an AI agent, start with transformer encoders and",
        user_input="and which algorithms are best for an ai agent",
        intent="conversation:general",
    )

    low = response.lower().strip()
    assert not low.endswith(" and")
    assert response.endswith(".")


def test_deterministic_knowledge_fallback_handles_ai_agent_algorithm_question():
    alice = ALICE.__new__(ALICE)

    response = alice._deterministic_knowledge_fallback(
        "and which algorithms are best for an ai agent",
        "conversation:general",
    )

    assert response is not None
    low = response.lower()
    assert "transformer" in low
    assert "retrieval" in low
    assert "verification" in low


def test_executive_controller_marks_difference_prompt_as_answerable_direct_question():
    controller = ExecutiveController()

    assert controller._is_answerability_direct_question(
        "i want to know the difference between the agentic ai and generative ai"
    ) is True


def test_safe_llm_failure_recovery_prefers_llm_retry_before_deterministic_fallback():
    alice = ALICE.__new__(ALICE)
    alice._internal_reasoning_state = {
        "confidence": 0.84,
        "response_plan": {"strategy": "answer_directly"},
    }

    calls = {"deterministic": 0}
    alice._retry_llm_answer_after_failure = lambda **_kwargs: "Retry answer from ollama."

    def _deterministic(*_args, **_kwargs):
        calls["deterministic"] += 1
        return "Deterministic fallback"

    alice._deterministic_fallback_once = _deterministic

    response = alice._safe_llm_failure_response(
        user_input="i want to know the difference between the agentic ai and generative ai",
        intent="conversation:goal_statement",
        llm_response=SimpleNamespace(success=False, response="", error="timeout"),
    )

    assert response == "Retry answer from ollama."
    assert calls["deterministic"] == 0


def test_deterministic_knowledge_fallback_handles_practical_agentic_framework_request():
    alice = ALICE.__new__(ALICE)

    response = alice._deterministic_knowledge_fallback(
        EXACT_FRAMEWORKS_PROMPT,
        "conversation:general",
    )

    assert response is not None
    low = response.lower()
    assert "langchain" in low
    assert "crewai" in low
    assert "react" in low
    assert "dennett" not in low
    assert "integrated information theory" not in low


def test_deterministic_knowledge_fallback_handles_existing_framework_inventory_prompt():
    alice = ALICE.__new__(ALICE)

    response = alice._deterministic_knowledge_fallback(
        "explain to me each existing framework",
        "conversation:question",
    )

    assert response is not None
    low = response.lower()
    assert "runtime contract pipeline" in low
    assert "memory framework" in low
    assert "policy and verification framework" in low


def test_semantic_fidelity_guard_rejects_theoretical_drift_for_practical_framework_prompt():
    controller = ExecutiveController()
    bad = (
        "A useful autonomy framework is Dennett's Intentional Stance and Tononi's IIT theory of consciousness."
    )

    evaluation = controller.evaluate_response(
        user_input=EXACT_FRAMEWORKS_PROMPT,
        intent="conversation:general",
        response=bad,
        route="llm",
        context={},
    )

    assert evaluation["accepted"] is False
    assert evaluation["reason"] == "semantic_drift_theoretical_domain"


def test_agent_algorithm_question_detector_matches_user_style_query():
    alice = ALICE.__new__(ALICE)

    assert alice._is_agent_algorithm_question(
        "and which algorithms are best for an ai agent"
    ) is True


def test_fast_lane_sanitizer_removes_hype_opener_and_inline_breaks():
    alice = ALICE.__new__(ALICE)

    response = alice._sanitize_fast_lane_response(
        response=(
            "You're looking to dive deeper into machine learning! Some popular advanced algorithms include Gradient Boosting.\n"
            "XGBoost and LightGBM are also strong choices."
        ),
        user_input="and which algorithms are best for an ai agent",
        intent="conversation:general",
    )

    low = response.lower()
    assert "you're looking to dive deeper" not in low
    assert "xgboost" in low
    assert "\n" not in response


def test_teaching_request_prefers_structured_mode_over_deterministic_fallback():
    alice = ALICE.__new__(ALICE)

    deterministic = alice._deterministic_knowledge_fallback(
        "teach me about nlp",
        "conversation:help",
    )
    structured = alice._structured_teaching_mode_response(
        "teach me about nlp",
        "conversation:help",
    )

    assert deterministic is None
    assert structured is not None
    low = structured.lower()
    assert "learning outline" in low
    assert "foundations" in low
    assert "methods" in low


def test_llm_failure_recovery_keeps_answer_directly_for_understood_safe_goal():
    alice = ALICE.__new__(ALICE)
    alice._internal_reasoning_state = {
        "confidence": 0.81,
        "response_plan": {"strategy": "answer_directly"},
    }

    response = alice._safe_llm_failure_response(
        user_input="i want to learn abit more about nlp",
        intent="conversation:help",
        llm_response=SimpleNamespace(
            success=False,
            response="Primary generation route is unavailable right now.",
            error="Primary generation route is unavailable right now.",
        ),
    )

    low = response.lower()
    assert "clarify" not in low
    assert "nlp" in low
    assert "intent" in low
    assert "conversation flow" in low

    state = dict(getattr(alice, "_internal_reasoning_state", {}) or {})
    recovery = dict(state.get("failure_recovery", {}) or {})
    assert recovery.get("avoid_clarification") is True


def test_compose_understood_goal_recovery_avoids_fixed_fallback_message():
    alice = ALICE.__new__(ALICE)

    response = alice._compose_understood_goal_recovery(
        "teach me about nlp",
        "conversation:help",
    )

    assert response is not None
    low = response.lower()
    assert "i still understand your goal" not in low
    assert "nlp" in low


def test_goal_statement_fallback_preserves_agentic_learning_guidance():
    alice = ALICE.__new__(ALICE)
    alice.adaptive_response_style = None

    response = alice._fallback_from_intent(
        "conversation:goal_statement",
        None,
        user_input="i want to learn more about agentic ai",
    )

    low = str(response or "").lower()
    assert "cleaner restatement" not in low
    assert "better understand your question" not in low
    assert "could you please specify" not in low
    assert any(
        token in low
        for token in (
            "agentic",
            "agent",
            "planning",
            "tool",
            "framework",
            "loop",
            "ai",
        )
    )


def test_self_answer_first_gate_uses_structured_teaching_mode_for_teach_prompts():
    alice = ALICE.__new__(ALICE)

    gate = alice._self_answer_first_gate(
        user_input="teach me about nlp",
        intent="conversation:help",
        entities={},
        has_active_goal=False,
        has_explicit_action_cue=False,
    )

    assert gate.get("block_llm") is True
    assert gate.get("reason") == "structured_teaching_mode"
    response = str(gate.get("response") or "")
    assert "Learning outline" in response


def test_self_answer_first_override_contract_respects_executive_llm_authority():
    alice = ALICE.__new__(ALICE)

    assert alice._self_answer_first_can_override("answer_direct") is True
    assert alice._self_answer_first_can_override("use_llm") is False
    assert alice._self_answer_first_can_override("ask_clarification") is False


def test_self_answer_first_gate_uses_non_structured_answer_for_learn_more_agentic_prompt():
    alice = ALICE.__new__(ALICE)

    gate = alice._self_answer_first_gate(
        user_input="i want to learn more about agentic ai",
        intent="conversation:goal_statement",
        entities={
            "goal": "learn more about agentic ai",
            "user_goal": "learn more about agentic ai",
        },
        has_active_goal=False,
        has_explicit_action_cue=False,
    )

    assert gate.get("block_llm") is True
    assert gate.get("reason") in {
        "deterministic_knowledge_fallback",
        "native_conceptual_answer",
    }
    response = str(gate.get("response") or "").lower()
    assert "learning outline" not in response
    assert "clarify" not in response
    assert any(
        token in response
        for token in (
            "agentic",
            "goal",
            "plan",
            "loop",
            "autonomy",
            "tool",
        )
    )


def test_native_scaffold_handles_simple_conversation_openers_without_llm():
    alice = ALICE.__new__(ALICE)

    assert alice._native_scaffold_response("how are you?", "status_inquiry") is not None
    assert alice._native_scaffold_response("how are you?", "conversation:general") is not None
    assert alice._native_scaffold_response("hello", "conversation:general") is not None
    assert alice._native_scaffold_response("can you help me?", "conversation:help") is None
    assert alice._native_scaffold_response("thanks", "conversation:general") is not None


def test_native_scaffold_does_not_flatten_beginner_explanation_help_request():
    alice = ALICE.__new__(ALICE)

    response = alice._native_scaffold_response(
        "i am beginner so i want an explanation",
        "conversation:help",
    )

    assert response is None


def test_native_scaffold_does_not_flatten_detailed_help_issue_report():
    alice = ALICE.__new__(ALICE)
    detailed = "my ai is not able to correctly give me some informations or it gets the intent wrong"

    assert alice._native_scaffold_response(detailed, "conversation:help") is None


def test_native_scaffold_disallowed_for_rich_conceptual_prompt():
    alice = ALICE.__new__(ALICE)

    response = alice._native_scaffold_response(
        EXACT_LOG_PROMPT,
        "conversation:clarification_needed",
    )

    assert response is None


def test_self_answer_gate_prefers_native_conceptual_for_rich_prompt():
    alice = ALICE.__new__(ALICE)

    gate = alice._self_answer_first_gate(
        user_input=EXACT_LOG_PROMPT,
        intent="conversation:clarification_needed",
        entities={},
        has_active_goal=False,
        has_explicit_action_cue=False,
    )

    assert gate.get("block_llm") is True
    assert gate.get("reason") == "native_conceptual_answer"
    response = str(gate.get("response") or "").lower()
    assert "real-world" in response or "real world" in response
    assert "architecture" in response
    assert any(
        token in response
        for token in ("foundations", "memory", "planning", "autonomy")
    )


def test_exact_fictional_inventor_prompt_blocks_scaffold_and_returns_direct_architecture_answer():
    alice = ALICE.__new__(ALICE)

    scaffold = alice._native_scaffold_response(
        EXACT_FICTIONAL_INVENTOR_PROMPT,
        "conversation:clarification_needed",
    )
    assert scaffold is None

    gate = alice._self_answer_first_gate(
        user_input=EXACT_FICTIONAL_INVENTOR_PROMPT,
        intent="conversation:clarification_needed",
        entities={},
        has_active_goal=False,
        has_explicit_action_cue=False,
    )

    assert gate.get("block_llm") is True
    assert gate.get("reason") == "native_conceptual_answer"
    response = str(gate.get("response") or "").lower()
    assert "language" in response and "understanding" in response
    assert "memory" in response
    assert "planning" in response
    assert "tool" in response
    assert "verification" in response
    assert "bounded autonomy" in response


def test_exact_create_prompt_blocks_scaffold_and_returns_direct_architecture_answer():
    alice = ALICE.__new__(ALICE)

    scaffold = alice._native_scaffold_response(
        EXACT_CREATE_PROMPT,
        "conversation:clarification_needed",
    )
    assert scaffold is None

    gate = alice._self_answer_first_gate(
        user_input=EXACT_CREATE_PROMPT,
        intent="conversation:clarification_needed",
        entities={},
        has_active_goal=False,
        has_explicit_action_cue=False,
    )

    assert gate.get("block_llm") is True
    assert gate.get("reason") == "native_conceptual_answer"
    response = str(gate.get("response") or "").lower()
    assert "language" in response and "understanding" in response
    assert "memory" in response
    assert "planning" in response
    assert "tool" in response
    assert "verification" in response
    assert "bounded autonomy" in response


def test_conceptual_build_prompt_does_not_reuse_goal_intent():
    alice = ALICE.__new__(ALICE)

    should_reuse = alice._should_reuse_goal_intent(
        EXACT_CREATE_PROMPT,
        "conversation clarification for route choice",
    )

    assert should_reuse is False


def test_goal_statement_promotion_enriches_entities_and_intent():
    alice = ALICE.__new__(ALICE)
    text = "I want to build an agent and not just a chatbot"

    intent, entities, confidence = alice._promote_goal_statement_intent(
        user_input=text,
        intent="conversation:general",
        entities={},
        intent_confidence=0.41,
    )

    assert intent == "conversation:goal_statement"
    assert confidence >= 0.84
    assert entities.get("goal")
    assert entities.get("user_goal")
    assert entities.get("project_direction")


def test_native_scaffold_goal_statement_returns_alignment_response():
    alice = ALICE.__new__(ALICE)
    text = "I want to make Alice think in steps and become more autonomous"

    response = alice._native_scaffold_response(text, "conversation:goal_statement")

    assert response is not None
    low = response.lower()
    assert "agent behavior" in low
    assert "planning" in low
    assert "persistent memory" in low
    assert "tool execution" in low


def test_execution_mode_prefers_conversational_fast_lane_for_conceptual_design_prompt():
    alice = ALICE.__new__(ALICE)
    prompt = "how can i design a real-world ai assistant architecture with today's technology and no fiction"

    mode, reason = alice._select_execution_mode(
        user_input=prompt,
        intent="learning:system_design",
        intent_confidence=0.84,
        has_active_goal=False,
        has_explicit_action_cue=False,
        force_plugins_for_notes=False,
        pending_action=None,
    )

    assert mode == "conversational_intelligence"
    assert reason == "rich_conceptual_fast_lane"


def test_execution_mode_uses_operator_for_explicit_action_turn():
    alice = ALICE.__new__(ALICE)

    mode, reason = alice._select_execution_mode(
        user_input="delete note 2 from my notebook",
        intent="notes:delete",
        intent_confidence=0.93,
        has_active_goal=False,
        has_explicit_action_cue=True,
        force_plugins_for_notes=False,
        pending_action=None,
    )

    assert mode == "operator"
    assert reason == "explicit_action_cue"


def test_execution_mode_locks_meta_question_to_conversational_lane_even_with_action_verbs():
    alice = ALICE.__new__(ALICE)

    mode, reason = alice._select_execution_mode(
        user_input="you are an ai how do you read a book or listen to calming music?",
        intent="conversation:meta_question",
        intent_confidence=0.92,
        has_active_goal=False,
        has_explicit_action_cue=True,
        force_plugins_for_notes=False,
        pending_action=None,
    )

    assert mode == "conversational_intelligence"
    assert reason == "meta_question_conversation_lock"


def test_conversational_fast_lane_detector_allows_brainstorming_without_tools():
    alice = ALICE.__new__(ALICE)

    is_fast_lane = alice._is_conversational_fast_lane_turn(
        user_input="can we brainstorm design options for an assistant memory strategy",
        intent="conversation:question",
        intent_confidence=0.76,
        has_active_goal=False,
        has_explicit_action_cue=False,
        force_plugins_for_notes=False,
        pending_action=None,
    )

    assert is_fast_lane is True


def test_recovery_seed_for_ai_project_prompt_is_natural_not_debug_key_value():
    alice = ALICE.__new__(ALICE)

    seed = alice._build_recovery_answer_seed(
        "i want to work on an ai project",
        "conversation:help",
    )

    low = seed.lower()
    assert "keywords=" not in low
    assert "user_goal=" not in low
    assert "intent=" not in low
    assert "prototype" in low


def test_distributed_cache_disabled_for_learning_prompts():
    alice = ALICE.__new__(ALICE)

    assert alice._should_use_distributed_response_cache("i want to learn about nlp") is False
    assert alice._should_use_distributed_response_cache("explain transformer embeddings") is False
    assert alice._should_use_distributed_response_cache("open notepad") is True


def test_execution_mode_prefers_fast_lane_for_active_goal_dialogue_turn():
    alice = ALICE.__new__(ALICE)

    mode, reason = alice._select_execution_mode(
        user_input="i want to learn about nlp",
        intent="conversation:help",
        intent_confidence=0.70,
        has_active_goal=True,
        has_explicit_action_cue=False,
        force_plugins_for_notes=False,
        pending_action=None,
    )

    assert mode == "conversational_intelligence"
    assert reason == "active_goal_dialogue_fast_lane"


def test_deterministic_knowledge_fallback_varies_across_repeated_turns():
    alice = ALICE.__new__(ALICE)

    first = alice._deterministic_knowledge_fallback(
        "i want to learn about nlp",
        "conversation:help",
    )
    second = alice._deterministic_knowledge_fallback(
        "i want to learn about nlp",
        "conversation:help",
    )

    assert first is not None
    assert second is not None
    assert first != second


def test_action_cue_detector_ignores_conceptual_build_prompt():
    alice = ALICE.__new__(ALICE)

    assert (
        alice._has_explicit_action_cue(
            "how can i create an ai like assistant with todays technology and no fiction"
        )
        is False
    )


def test_execution_mode_keeps_conceptual_create_prompt_in_conversational_mode():
    alice = ALICE.__new__(ALICE)

    mode, reason = alice._select_execution_mode(
        user_input="how can i create an ai like assistant with today's technology and no fiction",
        intent="conversation:help",
        intent_confidence=0.71,
        has_active_goal=False,
        has_explicit_action_cue=True,
        force_plugins_for_notes=False,
        pending_action=None,
    )

    assert mode == "conversational_intelligence"
    assert reason == "rich_conceptual_fast_lane"


def test_should_use_fast_llm_lane_true_for_non_action_learning_turn():
    alice = ALICE.__new__(ALICE)

    should_fast_lane = alice._should_use_fast_llm_lane(
        user_input="i want to learn about nlp",
        user_input_processed="i want to learn about nlp",
        intent="conversation:help",
        intent_confidence=0.70,
        entities={},
        has_active_goal=False,
        pending_action=None,
        plugin_scores={"conversation": 0.72, "notes": 0.22},
    )

    assert should_fast_lane is True


def test_should_use_fast_llm_lane_false_for_strong_tool_evidence():
    alice = ALICE.__new__(ALICE)

    should_fast_lane = alice._should_use_fast_llm_lane(
        user_input="show my calendar for today",
        user_input_processed="show my calendar for today",
        intent="conversation:help",
        intent_confidence=0.58,
        entities={},
        has_active_goal=False,
        pending_action=None,
        plugin_scores={"calendar": 0.93, "conversation": 0.31},
    )

    assert should_fast_lane is False


def test_fast_lane_sanitizer_removes_meta_training_and_history_claims():
    alice = ALICE.__new__(ALICE)

    noisy = (
        "NLP is fascinating. I've been trained on many conversations about it. "
        "(By the way, I've been keeping track of our conversation history. "
        "We were discussing machine learning last time.) "
        "Would you like practical applications or theory first?"
    )

    cleaned = alice._sanitize_fast_lane_response(
        response=noisy,
        user_input="i want to learn more about nlp",
        intent="conversation:help",
    )

    low = cleaned.lower()
    assert "trained" not in low
    assert "conversation history" not in low
    assert "last time" not in low


def test_clamp_final_response_fast_lane_enforces_shorter_cap():
    alice = ALICE.__new__(ALICE)

    long_text = " ".join(["nlp" for _ in range(500)])
    clamped = alice._clamp_final_response(
        long_text,
        tone="helpful",
        response_type="general_response",
        route="fast_llm_lane",
        user_input="i want to learn more about nlp",
    )

    assert len(clamped) <= 700


def test_clamp_final_response_project_ideation_meta_leak_uses_guidance():
    alice = ALICE.__new__(ALICE)

    leaked = "analysis: project_ideation context: user wants an ai agent plan: ask for exact result"
    clamped = alice._clamp_final_response(
        leaked,
        tone="helpful",
        response_type="general_response",
        route="generation",
        user_input="let's create an ai project, an ai agent",
    )

    low = clamped.lower()
    assert "i can help with that. tell me the exact result you want." not in low
    assert "strong place to start" in low
    assert "focus first on memory, tool-use, or conversational quality" in low


def test_clamp_final_response_rewrites_anthropomorphic_routine_claims():
    alice = ALICE.__new__(ALICE)

    leaked = "When I unwind, I usually like to read a book or listen to calming music."
    clamped = alice._clamp_final_response(
        leaked,
        tone="helpful",
        response_type="general_response",
        route="llm",
        user_input="just got home from work and im beat",
    )

    low = clamped.lower()
    assert "when i unwind" not in low
    assert "i usually like to" not in low
    assert "a practical option is to" in low


def test_contract_respond_stage_replaces_stale_clarification_scaffold_for_project_ideation() -> None:
    alice = ALICE.__new__(ALICE)

    repaired = alice._contract_respond_stage(
        user_input="i want to create an agentic local ai agent",
        candidate="I can help with that. Give me one concrete detail and I will answer directly.",
        reasoning_output=SimpleNamespace(intent="conversation:clarification_needed"),
        tool_results={},
    )

    low = repaired.lower()
    assert "one concrete detail" not in low
    assert "strong place to start" in low
    assert "focus first on memory, tool-use, or conversational quality" in low


def test_deterministic_framework_fallback_handles_short_agentic_prompt() -> None:
    alice = ALICE.__new__(ALICE)

    response = alice._deterministic_knowledge_fallback(
        "agentic ai frameworks",
        "conversation:help",
    )

    assert response is not None
    low = response.lower()
    assert "langgraph" in low
    assert "langchain" in low
    assert "crewai" in low
    assert "plan-execute-verify" in low
    assert "continue, retry, ask, or escalate" in low
    assert "consciousness" not in low
    assert "dennett" not in low
    assert "iit" not in low


def test_safe_llm_failure_recovery_records_explicit_answer_path_failure_reason() -> None:
    alice = ALICE.__new__(ALICE)
    alice._internal_reasoning_state = {
        "confidence": 0.78,
        "response_plan": {"strategy": "answer_directly"},
    }

    response = alice._safe_llm_failure_response(
        user_input="what's the difference between an ai agent and an ai assistant?",
        intent="conversation:help",
        llm_response=SimpleNamespace(
            success=False,
            response="Primary generation route is unavailable right now.",
            error="Primary generation route is unavailable right now.",
        ),
    )

    low = response.lower()
    assert "clarify" not in low
    state = dict(getattr(alice, "_internal_reasoning_state", {}) or {})
    recovery = dict(state.get("failure_recovery", {}) or {})
    taxonomy = dict(state.get("fallback_taxonomy", {}) or {})
    assert recovery.get("reason") == "llm_failed_after_answer_directly"
    assert taxonomy.get("reason") == "llm_failed_after_answer_directly"


def test_explicit_greeting_detector_and_native_greeting_path_for_hi_alice() -> None:
    alice = ALICE.__new__(ALICE)
    alice.phrasing_learner = None
    alice.context = None

    assert alice._is_explicit_greeting_input("hi alice") is True
    assert alice._native_scaffold_response("hi alice", "greeting") is not None


def test_explicit_greeting_detector_accepts_minor_name_typo() -> None:
    alice = ALICE.__new__(ALICE)

    assert alice._is_explicit_greeting_input("hi alioce") is True


def test_goal_statement_promotion_skips_social_greeting_turns() -> None:
    alice = ALICE.__new__(ALICE)

    intent, entities, confidence = alice._promote_goal_statement_intent(
        user_input="hi alioce",
        intent="greeting",
        entities={},
        intent_confidence=0.89,
    )

    assert intent == "greeting"
    assert entities == {}
    assert confidence == 0.89


def test_turn_outcome_resolution_requires_explicit_fallback_for_failed_llm() -> None:
    alice = ALICE.__new__(ALICE)

    route, success, recovered = alice._resolve_turn_success_and_route(
        default_route="unknown",
        plugin_result=None,
        llm_attempted=True,
        llm_generation_success=False,
        llm_fallback_applied=False,
    )
    assert route == "llm"
    assert success is False
    assert recovered is False

    route, success, recovered = alice._resolve_turn_success_and_route(
        default_route="unknown",
        plugin_result=None,
        llm_attempted=True,
        llm_generation_success=False,
        llm_fallback_applied=True,
    )
    assert route == "llm_fallback"
    assert success is True
    assert recovered is True


def test_llm_context_injects_recent_weather_only_for_weather_relevant_queries() -> None:
    class _Cache:
        def get(self, *_args, **_kwargs):
            return None

        def put(self, *_args, **_kwargs):
            return None

    class _Context:
        def get_personalization_context(self):
            return ""

    class _Memory:
        def recall_memory_weighted(self, *_args, **_kwargs):
            return []

        def get_context_for_llm(self, *_args, **_kwargs):
            return ""

    class _Plugins:
        def get_capabilities(self):
            return []

    class _LiveState:
        def freshest_weather_snapshot(self, **_kwargs):
            return {
                "source": "test",
                "data": {
                    "location": "Boston",
                    "temperature": 24,
                    "condition": "sunny",
                    "humidity": 48,
                    "wind_speed": 12,
                },
            }

    alice = ALICE.__new__(ALICE)
    alice.context_cache = _Cache()
    alice.context = _Context()
    alice.memory = _Memory()
    alice.plugins = _Plugins()
    alice.live_state_service = _LiveState()
    alice.self_reflection = SimpleNamespace(
        get_codebase_summary=lambda: {"base_path": ".", "total_files": 0}
    )
    alice._think = lambda *_args, **_kwargs: None
    alice._should_attach_goal_context = lambda *_args, **_kwargs: False
    alice.system_design_response_guard = None
    alice.episodic_memory_engine = None
    alice.semantic_memory_index = None
    alice.advanced_context = None
    alice.summarizer = None
    alice.conversation_state_tracker = None
    alice.goal_tracker = None
    alice.context_selector = None
    alice.reasoning_engine = None
    alice.world_state_memory = None
    alice._internal_reasoning_state = {}
    alice._get_recent_conversation_summary = lambda: ""
    alice._get_active_context = lambda: ""

    non_weather = alice._build_llm_context(
        "i want to learn more about agentic ai",
        "conversation:goal_statement",
        {},
        None,
    )
    assert "RECENT WEATHER:" not in non_weather

    weather = alice._build_llm_context(
        "should i bring an umbrella tomorrow?",
        "conversation:question",
        {},
        None,
    )
    assert "RECENT WEATHER:" in weather


def test_two_turn_llm_fallback_finalization_skips_publish_polish_and_runs_single_gate() -> None:
    alice = ALICE.__new__(ALICE)
    alice.phrasing_learner = None
    alice._internal_reasoning_state = {
        "confidence": 0.82,
        "response_plan": {"strategy": "answer_directly"},
    }

    greeting = alice._native_scaffold_response("hi alice", "greeting")
    assert greeting is not None

    alice._deterministic_fallback_once = (
        lambda _u, _i: "Agentic AI uses goals, planning, tool execution, and verification loops."
    )
    recovered = alice._safe_llm_failure_response(
        user_input="i want to learn more about agentic ai",
        intent="conversation:goal_statement",
        llm_response=SimpleNamespace(success=False, response="", error="timeout"),
    )

    low = recovered.lower()
    assert "clarify" not in low
    assert "exact result" not in low

    calls = {"publish": 0, "gate": 0}

    def _publish_stub(**kwargs):
        calls["publish"] += 1
        return kwargs["response"]

    def _gate_stub(**kwargs):
        calls["gate"] += 1
        return kwargs["response"]

    alice._publish_with_fast_llm_style = _publish_stub
    alice._apply_response_style_constraints = lambda text: text
    alice._prevent_unsolicited_summary = lambda **kwargs: kwargs["response"]
    alice._executive_apply_response_gate = _gate_stub

    final = alice._finalize_conversational_surface(
        user_input="i want to learn more about agentic ai",
        intent="conversation:goal_statement",
        response=recovered,
        route="llm_fallback",
        plugin_result=None,
        apply_publish_style=True,
    )

    assert calls["publish"] == 0
    assert calls["gate"] == 1
    assert str(final or "").strip() == str(recovered or "").strip()
