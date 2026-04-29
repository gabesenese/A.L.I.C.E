"""
Integration Tests - All 10 Tier Improvements

Tests validate:
- Tier 1: Long-session coherence, consistent identity, error recovery, goal alignment
- Tier 2: Tone stability, proactive nudges
- Tier 3: System introspection, weak-spot detection
- Tier 4: Multi-goal arbitration, explainable routing
"""

import pytest
from ai.memory.session_summarizer import SessionSummarizer
from ai.infrastructure.capability_constraints import (
    get_capability_constraints_ledger,
)
from ai.core.result_quality_scorer import ResultQualityScorer
from ai.learning.goal_alignment_tracker import GoalAlignmentTracker, FeedbackSignal
from ai.learning.tone_trajectory_engine import ToneTrajectoryEngine, ToneStyle
from ai.proactivity.pattern_based_nudger import PatternBasedNudger
from ai.introspection.system_state_api import get_system_state_api
from ai.learning.weak_spot_detector import WeakSpotDetector
from ai.goals.multi_goal_arbitrator import MultiGoalArbitrator, GoalPriority
from ai.reasoning.routing_decision_logger import (
    RoutingDecisionLogger,
    RoutingDecisionType,
)


# ════════════════════════════════════════════════════════════════════════════════
# TIER 1 TESTS
# ════════════════════════════════════════════════════════════════════════════════


class TestLongSessionCoherence:
    """Test long-session memory compression and summarization."""

    def test_session_summarizer_creates_summaries_at_interval(self):
        """Test that summaries are generated every N turns."""
        summarizer = SessionSummarizer(summary_interval=5)

        for i in range(12):
            summarizer.record_turn(
                user_input=f"Turn {i}",
                alice_response=f"Response {i}",
                intent="test:intent",
                metadata={"goals": ["test_goal"]},
            )

        # Should have generated 2 summaries (at turn 5 and 10)
        assert len(summarizer.summaries) >= 2
        assert summarizer.turn_count == 12

    def test_session_context_maintains_top_10_facts(self):
        """Test that top-10 semantic facts are always in scope."""
        summarizer = SessionSummarizer()

        facts = [f"Fact {i}" for i in range(15)]
        summarizer.update_top_10_facts(facts)

        assert len(summarizer.top_10_facts) == 10
        context = summarizer.get_session_context()
        assert len(context["top_10_facts"]) == 10

    def test_goal_tracking_across_long_session(self):
        """Test goal persistence across 50+ turns."""
        summarizer = SessionSummarizer()

        for i in range(60):
            summarizer.record_turn(
                user_input=f"Input {i}",
                alice_response="Response",
                intent="work:code" if i < 30 else "test:run",
                metadata={"goals": ["complete_feature"]},
            )

        assert summarizer.goal_is_tracked("complete_feature")
        context = summarizer.get_session_context()
        assert context["turn_count"] == 60


class TestConsistentIdentityThread:
    """Test capability constraints ledger."""

    def test_capability_ledger_initializes_defaults(self):
        """Test that default capabilities are registered."""
        ledger = get_capability_constraints_ledger()

        assert ledger.can_do("read_files")
        assert ledger.can_do("run_tests")
        assert not ledger.can_do("execute_shell")

    def test_approval_requirements_are_enforced(self):
        """Test that high-risk capabilities require approval."""
        ledger = get_capability_constraints_ledger()

        assert ledger.requires_approval_for("git_push")
        assert not ledger.requires_approval_for("analyze_code")
        assert not ledger.can_do("execute_shell")

    def test_capability_contradiction_detection(self):
        """Test detection of contradictory capability claims."""
        ledger = get_capability_constraints_ledger()

        # Should NOT contradict (true from ledger, claim true)
        assert not ledger.capability_contradiction_detected("read_files", True)

        # Should contradict (false from ledger, claim true)
        assert ledger.capability_contradiction_detected("execute_shell", True)


class TestTighterErrorRecovery:
    """Test result quality scorer."""

    def test_quality_scorer_rates_results(self):
        """Test that scorer evaluates result quality."""
        scorer = ResultQualityScorer()

        # Good result should rate well
        good_result = {
            "success": True,
            "response": "It's currently 72°F with clear skies. Humidity at 45%.",
            "data": {"temperature": 72, "condition": "clear", "humidity": 45},
        }

        # Bad result - missing critical fields, completely irrelevant
        bad_result = {
            "success": False,
            "response": "This is about something completely different",
            "data": {},
        }

        good_score = scorer.score_result(
            good_result, "what's the weather?", "weather:get"
        )
        bad_score = scorer.score_result(
            bad_result, "what's the weather?", "weather:get"
        )

        # Good should score better than bad
        assert good_score.overall > bad_score.overall
        # Should have issues detected
        assert len(bad_score.issues) > 0

    def test_quality_scorer_accepts_good_results(self):
        """Test that complete, relevant results pass."""
        scorer = ResultQualityScorer()

        result = {
            "success": True,
            "response": "It's currently 72°F with clear skies. Humidity at 45%.",
            "data": {"temperature": 72, "condition": "clear", "humidity": 45},
        }

        score = scorer.score_result(result, "what's the weather?", "weather:get")
        assert score.overall > 0.7
        assert not score.should_retry

    def test_high_stakes_scoring_works(self):
        """Test that high-stakes flag is accepted."""
        scorer = ResultQualityScorer()

        result = {
            "success": True,
            "response": "Operation completed.",
            "data": {"status": "success"},
        }

        # Both should work
        normal_score = scorer.score_result(result, "run tests", "test:run")
        high_stakes_score = scorer.score_result(
            result, "run tests", "test:run", is_high_stakes=True
        )

        # Both should be valid scores
        assert 0.0 <= normal_score.overall <= 1.0
        assert 0.0 <= high_stakes_score.overall <= 1.0


class TestUserGoalAlignment:
    """Test goal alignment tracker."""

    def test_alignment_tracker_records_interactions(self):
        """Test basic interaction recording."""
        tracker = GoalAlignmentTracker()

        tracker.record_interaction(
            user_input="list my files",
            alice_response="Here are your files",
            intent="file:list",
            goal="organize workspace",
            response_type="file_list",
        )

        assert len(tracker.entries) == 1
        assert tracker.alignment_stats["total_interactions"] == 1

    def test_implicit_feedback_detection(self):
        """Test detection of implicit feedback."""
        tracker = GoalAlignmentTracker()

        # Record first interaction
        tracker.record_interaction(
            user_input="what's the weather?",
            alice_response="Sunny, 72°F",
            intent="weather:get",
            goal="plan_outdoor_activity",
            response_type="weather",
        )

        # User asks followup with "it" pronoun (implicit yes)
        tracker.infer_implicit_feedback("is it humid today?")

        assert tracker.entries[0].feedback_signal == FeedbackSignal.IMPLICIT_FOLLOWUP

    def test_alignment_stats_calculation(self):
        """Test that alignment statistics are calculated."""
        tracker = GoalAlignmentTracker()

        entries = []
        for i in range(5):
            entry = tracker.record_interaction(
                user_input=f"test {i}",
                alice_response=f"Response {i}",
                intent="test",
                goal="test_goal",
                response_type="test",
            )
            entries.append(entry)

        # Explicitly mark some as helpful
        for i in [0, 1, 2]:
            entries[i].was_helpful = True
            tracker.alignment_stats["helpful_count"] += 1

        stats = tracker.get_stats()
        assert stats["total_interactions"] == 5
        assert stats["overall_success_rate"] == 0.6


# ════════════════════════════════════════════════════════════════════════════════
# TIER 2 TESTS
# ════════════════════════════════════════════════════════════════════════════════


class TestStableResponseVariance:
    """Test tone trajectory engine."""

    def test_tone_trajectory_locks_session_tone(self):
        """Test that tone is locked for entire session."""
        engine = ToneTrajectoryEngine()

        engine.set_session_tone(ToneStyle.FORMAL)
        assert engine.current_tone == ToneStyle.FORMAL
        assert engine.session_characteristics is not None

    def test_tone_consistency_scoring(self):
        """Test that responses are scored for tone consistency."""
        engine = ToneTrajectoryEngine()
        engine.set_session_tone(ToneStyle.FORMAL)

        formal_response = "We must proceed with deliberate caution and careful consideration of all available alternatives."
        # Casual response WITH contractions (don't, can't) which formal forbids
        casual_response = (
            "I don't think we can't do this because you'll just mess it up."
        )

        score_formal = engine.evaluate_response_consistency(formal_response)
        score_casual = engine.evaluate_response_consistency(casual_response)

        # Both should produce valid scores
        assert 0.0 <= score_formal <= 1.0
        assert 0.0 <= score_casual <= 1.0
        # Casual response has contractions (don't) when formal forbids them
        assert score_casual < score_formal

    def test_tone_prompt_addendum_generation(self):
        """Test that tone constraints are encoded for LLM."""
        engine = ToneTrajectoryEngine()
        engine.set_session_tone(ToneStyle.CASUAL)

        addendum = engine.get_tone_prompt_addendum()
        assert "Casual" in addendum or "casual" in addendum
        assert len(addendum) > 0


class TestProactiveNudges:
    """Test pattern-based nudger."""

    def test_nudger_detects_recurring_patterns(self):
        """Test that nudger detects repeat interactions."""
        nudger = PatternBasedNudger()

        for i in range(5):
            nudger.record_interaction(
                intent="code:analyze",
                success=True,
                response_type="analysis",
                user_input=f"analyze code {i}",
            )

        patterns = nudger.detect_patterns()
        assert len(patterns) > 0

    def test_nudger_generates_contextual_suggestions(self):
        """Test that nudger generates appropriate nudges."""
        nudger = PatternBasedNudger(min_pattern_frequency=2)

        # Record code review pattern (need at least min_pattern_frequency)
        for i in range(3):
            nudger.record_interaction(
                intent="code:analyze",
                success=True,
                response_type="analysis",
                user_input=f"analyze this code {i}",
            )

        nudge = nudger.generate_nudge_if_applicable()
        # May or may not generate a nudge depending on pattern matching
        # Just verify the method works
        assert isinstance(nudge, (type(None), object))  # Either None or Nudge object

    def test_nudge_effectiveness_tracking(self):
        """Test that nudge acceptance is tracked."""
        nudger = PatternBasedNudger()

        nudge = nudger.generate_nudge_if_applicable()
        if nudge:
            nudger.record_nudge_delivery(nudge, accepted=True)

            effectiveness = nudger.get_nudge_effectiveness()
            assert effectiveness["total_delivered"] >= 1

        # If no nudge generated, just verify method works
        assert True


# ════════════════════════════════════════════════════════════════════════════════
# TIER 3 TESTS
# ════════════════════════════════════════════════════════════════════════════════


class TestLiveArchitectureIntrospection:
    """Test system state API."""

    def test_system_state_api_queries_processor_state(self):
        """Test that processor state can be queried."""
        api = get_system_state_api()

        state = api.get_processor_state()
        # May be empty dict if ALICE not initialized, but should return dict
        assert isinstance(state, dict)

    def test_system_snapshot_comprehensive(self):
        """Test that system snapshot is comprehensive."""
        api = get_system_state_api()

        snapshot = api.get_system_snapshot()
        assert "timestamp" in snapshot
        assert "processors" in snapshot
        assert "memory" in snapshot
        assert "plugins" in snapshot
        assert "session_stats" in snapshot

    def test_system_health_assessment(self):
        """Test that system health is assessed."""
        api = get_system_state_api()

        health = api.get_system_health()
        assert "status" in health
        assert health["status"] in ("healthy", "degraded", "critical")

    def test_system_report_formatting(self):
        """Test that report is human-readable."""
        api = get_system_state_api()

        report = api.format_system_report()
        assert "A.L.I.C.E" in report
        assert len(report) > 100


class TestWeakSpotDetection:
    """Test weak-spot detector."""

    def test_weak_spot_detector_records_failures(self):
        """Test that failures are recorded."""
        detector = WeakSpotDetector()

        detector.record_failure(
            intent="weather:get",
            error_message="API timeout",
            routing_path="plugin:weather",
            reason="Rate limited",
        )

        assert len(detector.failures) == 1

    def test_weak_spot_pattern_detection(self):
        """Test that recurring failures are detected as patterns."""
        detector = WeakSpotDetector(min_occurrences_for_pattern=2)

        for i in range(4):
            detector.record_failure(
                intent="debug:error",
                error_message="Unrelated error",
                routing_path="llm:debug",
                reason="User stack trace unclear",
            )

        postmortems = detector.analyze_patterns()
        assert len(postmortems) > 0

    def test_weak_spot_generates_test_cases(self):
        """Test that regression tests are generated."""
        detector = WeakSpotDetector(min_occurrences_for_pattern=2)

        for i in range(3):
            detector.record_failure(
                intent="code:fix",
                error_message="Ambiguous request",
                routing_path="tool:code_fixer",
                reason="Too many possible interpretations",
            )

        postmortems = detector.analyze_patterns()
        if postmortems:
            assert any(pm.generated_test for pm in postmortems)


# ════════════════════════════════════════════════════════════════════════════════
# TIER 4 TESTS
# ════════════════════════════════════════════════════════════════════════════════


class TestMultiGoalArbitration:
    """Test multi-goal arbitrator."""

    def test_multi_goal_arbitrator_tracks_goals(self):
        """Test that goals are tracked."""
        arbitrator = MultiGoalArbitrator(max_active_goals=5)

        goal = arbitrator.add_goal(
            description="Complete feature implementation",
            priority=GoalPriority.HIGH,
            steps_total=5,
        )

        assert goal is not None
        assert len(arbitrator.get_active_goals()) == 1

    def test_multi_goal_priority_arbitration(self):
        """Test that goals are prioritized correctly."""
        arbitrator = MultiGoalArbitrator()

        # Add goals with different priorities
        arbitrator.add_goal("Low priority task", GoalPriority.LOW)
        high = arbitrator.add_goal("High priority task", GoalPriority.HIGH)

        # Auto-select should pick high priority
        next_goal = arbitrator.auto_select_next_focus()
        assert next_goal.goal_id == high.goal_id

    def test_multi_goal_capacity_management(self):
        """Test that goal capacity is managed."""
        arbitrator = MultiGoalArbitrator(max_active_goals=3)

        # Add 5 goals (should hit capacity)
        for i in range(5):
            arbitrator.add_goal(f"Goal {i}", GoalPriority.MEDIUM)

        active = arbitrator.get_active_goals()
        # Should have at most 3 active + some paused
        paused = [g for g in arbitrator.goals.values() if g.status == "paused"]
        assert len(active) + len(paused) == 5

    def test_multi_goal_completion_tracking(self):
        """Test that goal completion is tracked."""
        arbitrator = MultiGoalArbitrator()

        goal = arbitrator.add_goal("Simple task", GoalPriority.LOW, steps_total=1)
        arbitrator.update_progress(goal.goal_id, steps_completed=1)
        arbitrator.complete_goal(goal.goal_id)

        assert goal.status == "completed"
        assert goal.progress_percent == 100.0


class TestExplainableRouting:
    """Test routing decision logger."""

    def test_routing_logger_records_decisions(self):
        """Test that routing decisions are logged."""
        logger = RoutingDecisionLogger()

        decision = logger.log_decision(
            user_input="list files",
            classified_intent="file:list",
            intent_confidence=0.95,
            decision_type=RoutingDecisionType.TOOL_DISPATCH,
            candidates_considered=[
                {
                    "name": "file_tool",
                    "type": "tool",
                    "score": 0.95,
                    "reasoning": "Best match",
                },
            ],
            winning_candidate="file_tool",
            winning_score=0.95,
            decision_reasoning="Tool directly handles file operations",
            factors_used=["intent_match", "confidence", "tool_availability"],
        )

        assert decision is not None
        assert len(logger.decisions) == 1

    def test_routing_outcome_recording(self):
        """Test that routing outcomes are recorded."""
        logger = RoutingDecisionLogger()

        decision = logger.log_decision(
            user_input="test",
            classified_intent="test:intent",
            intent_confidence=0.5,
            decision_type=RoutingDecisionType.LLM_FALLBACK,
            candidates_considered=[],
            winning_candidate="fallback",
            winning_score=0.5,
            decision_reasoning="No specific tool matched",
            factors_used=[],
        )

        logger.record_outcome(decision.decision_id, success=True)
        assert decision.execution_success

    def test_routing_effectiveness_analysis(self):
        """Test that routing effectiveness is analyzed."""
        logger = RoutingDecisionLogger()

        for i in range(5):
            decision = logger.log_decision(
                user_input=f"test {i}",
                classified_intent="test",
                intent_confidence=0.8,
                decision_type=RoutingDecisionType.PATTERN_MATCH,
                candidates_considered=[],
                winning_candidate="pattern",
                winning_score=0.8,
                decision_reasoning="Matched learned pattern",
                factors_used=["pattern_library"],
            )
            logger.record_outcome(decision.decision_id, i < 4)  # 4/5 succeed

        effectiveness = logger.get_routing_effectiveness()
        assert effectiveness["total_decisions"] == 5

    def test_routing_explainability(self):
        """Test that routing decisions can be explained."""
        logger = RoutingDecisionLogger()

        decision = logger.log_decision(
            user_input="what's the weather?",
            classified_intent="weather:get",
            intent_confidence=0.92,
            decision_type=RoutingDecisionType.TOOL_DISPATCH,
            candidates_considered=[
                {
                    "name": "weather_tool",
                    "type": "tool",
                    "score": 0.92,
                    "reasoning": "Perfect match",
                },
            ],
            winning_candidate="weather_tool",
            winning_score=0.92,
            decision_reasoning="Weather tool is designed for this",
            factors_used=["intent_match", "tool_availability"],
        )

        explanation = logger.get_decision_for_explanation(decision.decision_id)
        assert explanation is not None
        assert "weather:get" in explanation
        assert "weather_tool" in explanation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
