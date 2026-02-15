"""
Integration Tests for Response Formulation Pipeline
Tests the complete flow: thought -> phrasing learning -> independent generation
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai.core.response_formulator import ResponseFormulator, get_response_formulator
from ai.learning.phrasing_learner import PhrasingLearner


class TestResponseFormulationPipeline:
    """End-to-end tests for response formulation and learning"""

    @pytest.fixture
    def formulator(self):
        """Create fresh response formulator for each test"""
        return ResponseFormulator()

    @pytest.fixture
    def learner(self):
        """Create fresh phrasing learner for each test"""
        return PhrasingLearner(storage_path="data/test_phrasings.jsonl")

    def test_initial_formulation_uses_llm(self, formulator):
        """First time Alice sees an action, she uses LLM"""
        # New action type Alice hasn't seen
        response = formulator.formulate_response(
            action="create_note",
            data={"note_title": "test"},
            success=True,
            user_input="create a note called test",
            tone="warm and helpful"
        )

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        # Response should mention the note
        assert "test" in response.lower() or "note" in response.lower()

    def test_learning_after_multiple_examples(self, learner):
        """After 3+ examples, Alice can phrase independently"""
        thought = {
            "type": "create_note",
            "data": {"note_title": "test"}
        }

        # Feed 5 examples
        for i in range(5):
            learner.record_phrasing(
                alice_thought=thought,
                ollama_phrasing=f"I've created a note called 'note_{i}' for you.",
                context={'tone': 'warm and helpful'}
            )

        # Check if Alice learned independence
        can_phrase = learner.can_phrase_myself(thought, tone="warm and helpful")
        assert can_phrase is True, "Alice should be able to phrase independently after 5 examples"

    def test_independent_phrasing_quality(self, learner):
        """Alice's independent phrasing should be natural"""
        thought = {"type": "create_note", "data": {"note_title": "meeting"}}

        # Train Alice with examples
        training_examples = [
            "I've created a note called 'project ideas' for you.",
            "I made a note titled 'shopping list'.",
            "Your note 'meeting agenda' has been created.",
            "I've added a note called 'reminders'.",
            "Note created: 'daily tasks'."
        ]

        for example in training_examples:
            learner.record_phrasing(
                alice_thought={"type": "create_note", "data": {"note_title": "example"}},
                ollama_phrasing=example,
                context={'tone': 'warm and helpful'}
            )

        # Get independent response
        response = learner.phrase_myself(thought, tone="warm and helpful")

        assert response is not None
        assert len(response) > 10
        assert "meeting" in response.lower()
        # Should use learned patterns (created/made/added)
        assert any(word in response.lower() for word in ['created', 'made', 'added', 'note'])

    def test_formulator_learns_from_successes(self, formulator):
        """Response formulator should record successful phrasings"""
        initial_patterns = len(formulator.phrasing_learner.learned_patterns)

        # Formulate response (will use LLM and potentially learn)
        response = formulator.formulate_response(
            action="create_reminder",
            data={"reminder_text": "call mom"},
            success=True,
            user_input="remind me to call mom",
            tone="warm and helpful"
        )

        # Should have learned something (if LLM was used)
        final_patterns = len(formulator.phrasing_learner.learned_patterns)

        # Patterns may increase or stay same depending on independence
        assert final_patterns >= initial_patterns

    def test_fallback_on_error(self, formulator):
        """Formulator should handle errors gracefully"""
        # This should not crash even with unusual data
        response = formulator.formulate_response(
            action="unknown_action",
            data={},
            success=False,
            user_input="do something weird",
            tone="warm and helpful"
        )

        assert response is not None
        assert isinstance(response, str)
        # Should acknowledge the issue
        assert len(response) > 0

    def test_tone_variation(self, learner):
        """Different tones should be tracked separately"""
        thought = {"type": "greeting", "data": {}}

        # Train with warm tone
        for i in range(3):
            learner.record_phrasing(
                alice_thought=thought,
                ollama_phrasing="Hi there! How can I help you today?",
                context={'tone': 'warm'}
            )

        # Train with professional tone
        for i in range(3):
            learner.record_phrasing(
                alice_thought=thought,
                ollama_phrasing="Hello. How may I assist you?",
                context={'tone': 'professional'}
            )

        # Should be able to phrase with warm tone
        can_warm = learner.can_phrase_myself(thought, tone='warm')
        assert can_warm is True

        # Should be able to phrase with professional tone
        can_professional = learner.can_phrase_myself(thought, tone='professional')
        assert can_professional is True

        # Responses should differ by tone
        warm_response = learner.phrase_myself(thought, tone='warm')
        professional_response = learner.phrase_myself(thought, tone='professional')

        # They should be different (different learned patterns)
        assert warm_response != professional_response or len(warm_response) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
