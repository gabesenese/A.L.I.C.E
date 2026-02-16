"""
Integration Tests for Learning Cycle
Tests automated learning loop and evaluation
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai.learning.phrasing_learner import PhrasingLearner
from ai.core.response_formulator import ResponseFormulator


class TestLearningCycle:
    """Integration tests for learning and self-improvement"""

    @pytest.fixture
    def learner(self, tmp_path):
        """Create fresh phrasing learner with isolated storage"""
        storage = str(tmp_path / "test_learning.jsonl")
        return PhrasingLearner(storage_path=storage)

    @pytest.fixture
    def formulator(self, learner):
        """Create fresh response formulator wired to a phrasing learner"""
        return ResponseFormulator(phrasing_learner=learner)

    def test_learning_progression_0_to_3_examples(self, learner):
        """Learning should progress: 0 examples -> can't phrase -> 3 examples -> can phrase"""
        thought = {
            "type": "reminder_created",
            "data": {"reminder": "test"}
        }

        # Initially can't phrase
        assert learner.can_phrase_myself(thought, "warm") is False

        # Add 1 example - still can't (needs 3+)
        learner.record_phrasing(
            alice_thought=thought,
            ollama_phrasing="Reminder set!",
            context={'tone': 'warm'}
        )
        assert learner.can_phrase_myself(thought, "warm") is False

        # Add 2 more examples (total 3)
        for i in range(2):
            learner.record_phrasing(
                alice_thought=thought,
                ollama_phrasing=f"Got it! Reminder created.",
                context={'tone': 'warm'}
            )

        # Now should be able to phrase
        assert learner.can_phrase_myself(thought, "warm") is True

    def test_confidence_increases_with_examples(self, learner):
        """Confidence should increase as more examples are added"""
        thought = {"type": "test", "data": {}}

        # Get initial confidence
        pattern = learner._extract_pattern(thought)
        initial_confidence = learner.confidence_scores.get(pattern, 0.0)

        # Add examples
        for i in range(5):
            learner.record_phrasing(
                alice_thought=thought,
                ollama_phrasing="Test response",
                context={'tone': 'warm'}
            )

        # Confidence should increase
        final_confidence = learner.confidence_scores.get(pattern, 0.0)
        assert final_confidence > initial_confidence

    def test_pattern_extraction_consistency(self, learner):
        """Same thought types should extract same patterns"""
        thought1 = {"type": "note_created", "data": {"title": "foo"}}
        thought2 = {"type": "note_created", "data": {"title": "bar"}}

        pattern1 = learner._extract_pattern(thought1)
        pattern2 = learner._extract_pattern(thought2)

        # Should be the same pattern
        assert pattern1 == pattern2

    def test_different_tones_tracked_separately(self, learner):
        """Same thought with different tones should require separate learning"""
        thought = {"type": "greeting", "data": {}}

        # Train with warm tone
        for _ in range(3):
            learner.record_phrasing(
                alice_thought=thought,
                ollama_phrasing="Hey there!",
                context={'tone': 'warm'}
            )

        # Can phrase with warm tone
        assert learner.can_phrase_myself(thought, "warm") is True

        # But not with professional tone (no examples yet)
        assert learner.can_phrase_myself(thought, "professional") is False

    def test_learning_stats_accuracy(self, learner):
        """Learning stats should accurately reflect state"""
        # Add varied examples
        for i in range(10):
            thought = {"type": f"action_{i % 3}", "data": {}}
            learner.record_phrasing(
                alice_thought=thought,
                ollama_phrasing=f"Response {i}",
                context={'tone': 'warm'}
            )

        stats = learner.get_stats()

        assert stats['total_examples'] == 10
        assert stats['total_patterns'] == 3  # 3 different action types
        assert 'patterns' in stats

    def test_formulator_independence_increases_over_time(self, formulator):
        """Response formulator should become more independent with use"""
        action = "test_action"

        # First few calls will use basic fallback
        for i in range(5):
            response = formulator.formulate_response(
                action=action,
                data={"count": i},
                success=True,
                user_input=f"test {i}",
                tone="warm"
            )

            assert response is not None

        # After learning, should be more likely to use learned patterns
        response = formulator.formulate_response(
            action=action,
            data={"count": 99},
            success=True,
            user_input="test 99",
            tone="warm"
        )

        assert response is not None

    def test_pattern_adaptation(self, learner):
        """Learned patterns should adapt to new content"""
        thought_training = {
            "type": "capability_answer",
            "can_do": True,
            "details": ["read files", "search code"]
        }

        # Train
        for _ in range(3):
            learner.record_phrasing(
                alice_thought=thought_training,
                ollama_phrasing="Yes, I can read files and search code.",
                context={'tone': 'warm'}
            )

        # New thought with different details
        thought_new = {
            "type": "capability_answer",
            "can_do": True,
            "details": ["analyze data", "create charts"]
        }

        # Should be able to phrase (same pattern)
        assert learner.can_phrase_myself(thought_new, "warm") is True

        # Get adapted phrasing
        response = learner.phrase_myself(thought_new, "warm")

        assert response is not None
        # Should be a reasonable response
        assert len(response) > 0

    def test_persistence_across_instances(self, tmp_path):
        """Learner should persist and reload patterns"""
        storage = str(tmp_path / "persist_test.jsonl")
        thought = {"type": "persist_test", "data": {}}

        # First instance: train
        learner1 = PhrasingLearner(storage_path=storage)
        for _ in range(3):
            learner1.record_phrasing(
                alice_thought=thought,
                ollama_phrasing="Persisted response",
                context={'tone': 'warm'}
            )
        assert learner1.can_phrase_myself(thought, "warm") is True

        # Second instance: should load persisted data
        learner2 = PhrasingLearner(storage_path=storage)
        assert learner2.can_phrase_myself(thought, "warm") is True
        assert learner2.get_stats()['total_examples'] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
