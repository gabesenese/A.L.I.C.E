"""
Integration tests for:
  - ResponsePlanner (type detection, strategy selection, prompt injection)
  - GoalTracker (persistence, drift correction, completion, alignment)
  - ResponseQualityTracker (quality metrics, failure classification)
  - ExecutiveController save/load weights
  - ConversationStateTracker drift correction
"""

import os
import json
import tempfile
import pytest

from ai.core.response_planner import ResponsePlanner, get_response_planner
from ai.core.goal_tracker import GoalTracker, get_goal_tracker
from ai.core.response_quality_tracker import (
    ResponseQualityTracker,
    FAILURE_NONE,
    FAILURE_OVERGENERALIZATION,
    FAILURE_WEAK_KNOWLEDGE,
    FAILURE_TOPIC_DRIFT,
    FAILURE_ROUTING_MISTAKE,
)
from ai.core.executive_controller import ExecutiveController
from ai.core.reflection_engine import ReflectionEngine
from ai.memory.conversation_state import ConversationStateTracker


# ===========================================================================
# ResponsePlanner
# ===========================================================================

class TestResponsePlanner:
    def setup_method(self):
        self.planner = ResponsePlanner()

    def _plan(self, user_input, intent="question:general", depth=0, goal=""):
        rs = {"depth_level": depth, "user_goal": goal, "confidence": 0.8, "conversation_goal": ""}
        cs = {"depth_level": depth, "user_goal": goal, "conversation_goal": ""}
        return self.planner.plan(
            user_input=user_input,
            intent=intent,
            reasoning_state=rs,
            conversation_state=cs,
        )

    def test_detects_debugging_type(self):
        plan = self._plan("there's a bug in my code")
        assert plan.response_type == "debugging"

    def test_detects_instruction_type(self):
        plan = self._plan("how to install Python step by step")
        assert plan.response_type == "instruction"

    def test_detects_planning_type(self):
        plan = self._plan("help me design the architecture")
        assert plan.response_type == "planning"

    def test_detects_conversational_type(self):
        plan = self._plan("hello, how are you?", intent="conversation:chitchat")
        assert plan.response_type == "conversational"

    def test_explanation_strategy_for_learning_goal(self):
        plan = self._plan("explain decorators", intent="learning:python", depth=1, goal="learn python")
        assert plan.strategy == "guided_explanation"

    def test_expand_strategy_at_deep_depth(self):
        plan = self._plan("explain decorators in depth", intent="learning:python", depth=4, goal="learn python")
        assert plan.strategy == "incremental_teaching"

    def test_plan_contains_required_sections_and_depth(self):
        plan = self._plan("steps to deploy flask", intent="question:general", depth=4)
        assert len(plan.required_sections) >= 2
        assert plan.plan_depth == 3

    def test_guiding_question_generation(self):
        rs = {"depth_level": 1, "user_goal": "learn python", "confidence": 0.3, "conversation_goal": "learning"}
        cs = {"depth_level": 1, "user_goal": "learn python", "conversation_goal": "learning"}
        plan = self.planner.plan(
            user_input="explain generators",
            intent="learning:python",
            reasoning_state=rs,
            conversation_state=cs,
        )
        q = self.planner.guiding_question(plan, "explain generators")
        assert len(q) > 10

    def test_outline_is_non_empty(self):
        plan = self._plan("fix this error")
        assert len(plan.outline) >= 2

    def test_goal_constraint_injected(self):
        plan = self._plan("how do I fix this bug", goal="fix authentication error")
        assert any("fix authentication error" in c for c in plan.constraints)

    def test_instruction_format_hint(self):
        plan = self._plan("steps to deploy a flask app")
        assert plan.format_hint == "numbered list"

    def test_prompt_injection_contains_type_and_strategy(self):
        plan = self._plan("explain async await")
        text = plan.to_prompt_injection()
        assert "type:" in text
        assert "strategy:" in text

    def test_singleton_returns_same_instance(self):
        a = get_response_planner()
        b = get_response_planner()
        assert a is b


# ===========================================================================
# GoalTracker
# ===========================================================================

class TestGoalTracker:
    def setup_method(self):
        self.tracker = GoalTracker()

    def _update(self, user_input="ok", response="here is the answer", user_goal="", conv_goal=""):
        self.tracker.update(
            user_input=user_input,
            response=response,
            user_goal=user_goal,
            conversation_goal=conv_goal,
            intent="question:general",
        )

    def test_topic_shift_creates_related_subgoal(self):
        self.tracker.update(
            user_input="teach me python",
            response="Starting with basics",
            user_goal="learn python",
            conversation_goal="learning",
            intent="learning:python",
            previous_topic="python basics",
            current_topic="python decorators",
        )
        status = self.tracker.get_status()
        assert status is not None
        assert len(status["subgoals"]) >= 1

    def test_bug_goal_checklist_progress(self):
        self._update(user_goal="fix login bug", response="Root cause is missing token. Fix by adding validation.")
        status = self.tracker.get_status()
        assert status is not None
        assert "completion_checklist" in status
        assert status["completion_checklist"].get("problem_identified") is True
        assert status["completion_checklist"].get("solution_given") is True

    def test_goal_set_on_first_update(self):
        self._update(user_goal="learn Python")
        assert self.tracker.get_status() is not None
        assert "python" in self.tracker.get_status()["goal_description"].lower()

    def test_goal_evolves_on_related_shift(self):
        self._update(user_goal="learn Python")
        self._update(user_goal="learn Python decorators")
        # Should still be one goal, not replaced
        status = self.tracker.get_status()
        assert "python" in status["goal_description"].lower()
        assert status["turns_since_set"] >= 1

    def test_subgoal_created_on_early_divergence(self):
        self._update(user_goal="learn Python")
        # Early turn (< 3), different topic → subgoal
        self._update(user_goal="fix authentication bug")
        status = self.tracker.get_status()
        assert len(status["subgoals"]) >= 1

    def test_goal_replaced_after_sustained_divergence(self):
        self._update(user_goal="learn Python")
        # Push turns_since_set above the subgoal threshold (>2)
        self._update(user_goal="learn Python")
        self._update(user_goal="learn Python")
        self._update(user_goal="learn Python")
        # Now diverge — should replace, not subgoal
        self._update(user_goal="book a flight to Paris")
        status = self.tracker.get_status()
        assert "paris" in status["goal_description"].lower() or "flight" in status["goal_description"].lower()

    def test_completion_detected_on_user_acknowledgement(self):
        self._update(user_goal="fix the login bug")
        self._update(user_input="thanks, that solved it!", user_goal="fix the login bug")
        assert self.tracker.is_goal_achieved()

    def test_goal_alignment_score_high_on_match(self):
        self._update(user_goal="learn Python async")
        score = self.tracker.goal_alignment_score("Python async await explained with examples")
        assert score > 0.20

    def test_goal_alignment_score_returns_one_when_no_goal(self):
        tracker = GoalTracker()
        assert tracker.goal_alignment_score("anything") == 1.0

    def test_prompt_injection_contains_goal(self):
        self._update(user_goal="debug Flask app")
        injection = self.tracker.get_goal_prompt_injection()
        assert "flask" in injection.lower() or "debug" in injection.lower()

    def test_next_step_suggestion_when_complete(self):
        self._update(user_goal="write unit tests")
        self._update(user_input="perfect thanks", user_goal="write unit tests")
        suggestion = self.tracker.get_next_step_suggestion()
        assert len(suggestion) > 0

    def test_singleton_returns_same_instance(self):
        a = get_goal_tracker()
        b = get_goal_tracker()
        assert a is b


# ===========================================================================
# ResponseQualityTracker
# ===========================================================================

class TestResponseQualityTracker:
    def setup_method(self):
        self.tracker = ResponseQualityTracker()

    def _track(self, user_input="what is Python?", response="Python is a language",
                intent="question:general", gate_accepted=True, reflection_score=0.75):
        return self.tracker.track_turn(
            user_input=user_input,
            response=response,
            intent=intent,
            gate_accepted=gate_accepted,
            reflection_score=reflection_score,
        )

    def test_successful_turn_returns_none_failure(self):
        q = self._track(gate_accepted=True, reflection_score=0.80)
        assert q.failure_type == FAILURE_NONE

    def test_overgeneralization_detected(self):
        bad_response = (
            "It depends. In general there are many factors. "
            "As an AI I cannot be determined to give a specific answer."
        )
        q = self._track(response=bad_response, gate_accepted=False, reflection_score=0.30)
        assert q.failure_type == FAILURE_OVERGENERALIZATION

    def test_weak_knowledge_detected(self):
        bad_response = "I'm not sure. Maybe possibly you could try. I don't know for certain."
        q = self._track(response=bad_response, gate_accepted=False, reflection_score=0.30)
        assert q.failure_type == FAILURE_WEAK_KNOWLEDGE

    def test_topic_drift_detected(self):
        q = self._track(
            user_input="how do I fix my car engine",
            response="The weather is nice today and flowers are blooming",
            gate_accepted=False,
            reflection_score=0.20,
        )
        assert q.failure_type == FAILURE_TOPIC_DRIFT

    def test_routing_mistake_detected(self):
        long_prose = " ".join(["word"] * 50)
        q = self._track(
            intent="notes:create_note",
            response=long_prose,
            gate_accepted=False,
            reflection_score=0.30,
        )
        assert q.failure_type == FAILURE_ROUTING_MISTAKE

    def test_quality_summary_accumulates_turns(self):
        for _ in range(5):
            self._track()
        summary = self.tracker.get_quality_summary()
        assert summary["turns_tracked"] == 5
        assert 0.0 <= summary["relevance_avg"] <= 1.0
        assert 0.0 <= summary["topic_adherence_avg"] <= 1.0
        assert 0.0 <= summary["adherence_avg"] <= 1.0
        assert 0.0 <= summary["gate_accept_rate"] <= 1.0

    def test_turn_quality_dict_contains_adherence(self):
        q = self._track()
        data = q.as_dict()
        assert "adherence" in data
        assert 0.0 <= data["adherence"] <= 1.0

    def test_topic_adherence_uses_topic_hint(self):
        q = self.tracker.track_turn(
            user_input="explain python generators",
            response="Generators in Python yield values lazily.",
            intent="learning:python",
            gate_accepted=True,
            topic_hint="python generators",
        )
        assert q.topic_adherence >= q.relevance

    def test_coherence_score_high_for_varied_response(self):
        q = self._track(
            response="Python is a high-level interpreted language known for readability and simplicity"
        )
        assert q.coherence > 0.5

    def test_verbosity_ideal_for_medium_length(self):
        medium = " ".join(["word"] * 80)
        q = self._track(response=medium)
        assert q.verbosity == 0.5

    def test_singleton_returns_same_instance(self):
        from ai.core.response_quality_tracker import get_response_quality_tracker
        a = get_response_quality_tracker()
        b = get_response_quality_tracker()
        assert a is b


# ===========================================================================
# ExecutiveController save/load weights
# ===========================================================================

class TestExecutiveWeightPersistence:
    def test_save_and_load_weights_roundtrip(self):
        ctrl = ExecutiveController()
        ctrl._routing_weights["llm"] = 1.3
        ctrl._routing_weights["tools"] = 0.7

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "weights.json")
            ctrl.save_weights(path)
            assert os.path.exists(path)

            ctrl2 = ExecutiveController()
            ctrl2.load_weights(path, decay=0.0)  # decay=0 → exact values
            assert abs(ctrl2._routing_weights["llm"] - 1.3) < 1e-6
            assert abs(ctrl2._routing_weights["tools"] - 0.7) < 1e-6

    def test_load_applies_decay_toward_neutral(self):
        ctrl = ExecutiveController()
        ctrl._routing_weights["llm"] = 1.4

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "weights.json")
            ctrl.save_weights(path)

            ctrl2 = ExecutiveController()
            ctrl2.load_weights(path, decay=0.10)
            # With decay=0.10, loaded 1.4 → 1.4 + (1.0-1.4)*0.10 = 1.4 - 0.04 = 1.36
            assert ctrl2._routing_weights["llm"] < 1.4
            assert ctrl2._routing_weights["llm"] > 1.0

    def test_load_missing_file_is_silent(self):
        ctrl = ExecutiveController()
        original = dict(ctrl._routing_weights)
        ctrl.load_weights("/nonexistent/path/weights.json")
        assert ctrl._routing_weights == original

    def test_save_creates_valid_json(self):
        ctrl = ExecutiveController()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "weights.json")
            ctrl.save_weights(path)
            with open(path) as f:
                data = json.load(f)
            assert isinstance(data, dict)
            assert "llm" in data


class TestExecutivePlanEnforcement:
    def test_gate_rejects_instruction_without_steps(self):
        ctrl = ExecutiveController()
        evaluation = ctrl.evaluate_response(
            user_input="give me step by step instructions to deploy",
            intent="question:general",
            response="You can deploy by using Docker and cloud tools.",
            route="llm",
            context={
                "response_plan": {
                    "response_type": "instruction",
                    "format_hint": "numbered list",
                    "required_sections": ["steps", "expected_result"],
                },
                "goal_alignment": 1.0,
            },
        )
        assert evaluation["accepted"] is False


class TestReflectionQualityFeedback:
    def test_reflection_adjusts_for_routing_mistake(self):
        engine = ReflectionEngine()
        res = engine.reflect(
            user_input="create a note",
            intent="notes:create_note",
            response="This is a long generic explanation that does not execute actions.",
            route="llm",
            gate_accepted=False,
            decision_scores={"llm": 0.6, "tools": 0.59},
            prior_confidence=0.5,
            quality_metrics={"topic_adherence": 0.2, "alignment": 0.4},
            failure_type="routing_mistake",
        )
        out = res.as_dict()
        assert out["routing_adjustments"].get("tools", 0.0) > 0


# ===========================================================================
# ConversationStateTracker drift correction
# ===========================================================================

class TestConversationStateDriftCorrection:
    def setup_method(self):
        self.tracker = ConversationStateTracker()

    def test_related_topic_shift_does_not_reset_chains(self):
        # Establish initial state with a question
        self.tracker.update_state(
            user_input="how do I learn Python",
            intent="learning:python",
            entities={"topic": "python"},
        )
        q_chain_before = list(self.tracker.get_state_summary()["question_chain"])
        # Related subtopic shift
        self.tracker.update_state(
            user_input="what about Python decorators",
            intent="learning:python",
            entities={"topic": "python decorators"},
        )
        state = self.tracker.get_state_summary()
        # Chains should NOT be empty (drift prevention kept them)
        assert len(state["question_chain"]) >= 1

    def test_unrelated_topic_shift_resets_chains(self):
        self.tracker.update_state(
            user_input="how do I learn Python",
            intent="learning:python",
            entities={"topic": "python"},
        )
        # Completely unrelated topic
        self.tracker.update_state(
            user_input="book me a flight to Tokyo",
            intent="calendar:create_event",
            entities={"topic": "flight"},
        )
        state = self.tracker.get_state_summary()
        # Chains must have reset (the Python question is gone)
        python_in_chain = any("python" in q.lower() for q in state["question_chain"])
        assert not python_in_chain

    def test_topic_similar_to_goal_helper(self):
        self.tracker.update_state(
            user_input="help me understand async programming",
            intent="learning:python",
            entities={"topic": "async programming"},
        )
        # Manually verify similarity check works
        assert self.tracker._topic_similar_to_goal("async await in Python")
        assert not self.tracker._topic_similar_to_goal("how to bake a cake")
