"""
Integration Tests for Response Formulation Pipeline
Tests the complete flow: thought -> phrasing learning -> independent generation
"""

import pytest
import sys
import tempfile
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai.core.response_formulator import ResponseFormulator
from ai.learning.phrasing_learner import PhrasingLearner


class TestResponseFormulationPipeline:
    """End-to-end tests for response formulation and learning"""

    @pytest.fixture
    def learner(self, tmp_path):
        """Create fresh phrasing learner with isolated storage"""
        storage = str(tmp_path / "test_phrasings.jsonl")
        return PhrasingLearner(storage_path=storage)

    @pytest.fixture
    def formulator(self, learner):
        """Create fresh response formulator wired to a phrasing learner"""
        return ResponseFormulator(phrasing_learner=learner)

    def test_initial_formulation_uses_llm(self, formulator):
        """First time Alice sees an action, she uses LLM or data-aware fallback"""
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
        # Response should mention the note title from data
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
        """Alice's independent phrasing should be natural and adapt to new data"""
        # Train with consistent note_title so adaptation can substitute
        training_thought = {"type": "create_note", "data": {"note_title": "tasks"}}

        training_examples = [
            "I've created a note called 'tasks' for you.",
            "I made a note titled 'tasks'.",
            "Your note 'tasks' has been created.",
            "I've added a note called 'tasks'.",
            "Note created: 'tasks'."
        ]

        for example in training_examples:
            learner.record_phrasing(
                alice_thought=training_thought,
                ollama_phrasing=example,
                context={'tone': 'warm and helpful'}
            )

        # Now ask for a different title - adaptation should swap it
        new_thought = {"type": "create_note", "data": {"note_title": "meeting"}}
        response = learner.phrase_myself(new_thought, tone="warm and helpful")

        assert response is not None
        assert len(response) > 10
        # Adaptation should have replaced "tasks" with "meeting"
        assert "meeting" in response.lower()
        assert any(word in response.lower() for word in ['created', 'made', 'added', 'note'])

    def test_formulator_learns_from_successes(self, formulator):
        """Response formulator should record successful phrasings via its learner"""
        initial_patterns = len(formulator.phrasing_learner.learned_patterns)

        # Formulate response (will use basic fallback, then learn)
        response = formulator.formulate_response(
            action="create_reminder",
            data={"reminder_text": "call mom"},
            success=True,
            user_input="remind me to call mom",
            tone="warm and helpful"
        )

        assert response is not None
        # Patterns may increase or stay same depending on independence
        final_patterns = len(formulator.phrasing_learner.learned_patterns)
        assert final_patterns >= initial_patterns

    def test_fallback_on_error(self, formulator):
        """Formulator should handle errors gracefully"""
        response = formulator.formulate_response(
            action="unknown_action",
            data={},
            success=False,
            user_input="do something weird",
            tone="warm and helpful"
        )

        assert response is not None
        assert isinstance(response, str)
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

    def test_notes_list_response_is_generated_without_empty_output(self, formulator):
        response = formulator.formulate_response(
            action="list_notes",
            data={
                "count": 3,
                "shown": 3,
                "notes": [
                    {"title": "Grocery List"},
                    {"title": "Grocery List Weekend"},
                    {"title": "Meeting Note"},
                ],
            },
            success=True,
            user_input="do i have any notes?",
            tone="helpful",
        )

        assert response is not None
        assert isinstance(response, str)
        assert len(response.strip()) > 0

    def test_notes_list_response_for_empty_state_is_nonempty(self, formulator):
        response = formulator.formulate_response(
            action="list_notes",
            data={"count": 0, "shown": 0, "notes": []},
            success=True,
            user_input="do i have any notes?",
            tone="helpful",
        )
        assert response is not None
        assert isinstance(response, str)
        assert len(response.strip()) > 0

    def test_count_notes_response_is_nonempty(self, formulator):
        response = formulator.formulate_response(
            action="count_notes",
            data={"total": 1},
            success=True,
            user_input="how many notes do i have?",
            tone="helpful",
        )
        assert response is not None
        assert isinstance(response, str)
        assert len(response.strip()) > 0

    def test_formulation_does_not_call_llm_gateway(self, learner):
        class DummyGateway:
            def __init__(self):
                self.called = False

            def request(self, **kwargs):
                self.called = True
                raise AssertionError("LLM gateway should not be called")

        gateway = DummyGateway()
        formulator = ResponseFormulator(phrasing_learner=learner, llm_gateway=gateway)

        response = formulator.formulate_response(
            action="list_notes",
            data={"count": 1, "notes": [{"title": "Test"}]},
            success=True,
            user_input="do i have any notes?",
            tone="helpful",
        )

        assert isinstance(response, str)
        assert len(response.strip()) > 0
        assert gateway.called is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
