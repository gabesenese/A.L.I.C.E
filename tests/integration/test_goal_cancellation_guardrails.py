"""
Integration tests for goal cancellation guardrails.
Ensures mixed conversational inputs are not short-circuited as pure cancellation.
"""

import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai.core.reasoning_engine import ActiveGoal, ReasoningEngine


class TestGoalCancellationGuardrails:
    @pytest.fixture
    def engine(self):
        return ReasoningEngine(user_name="Gabriel")

    def test_pure_cancel_still_cancels_goal(self, engine):
        engine.set_goal(
            ActiveGoal(
                goal_id="goal_1",
                description="check weather",
                intent="weather:forecast",
                entities={},
            )
        )

        result = engine.resolve_goal("never mind", "conversation:ack", {})

        assert result.cancelled is True
        assert result.intent == "conversation:ack"
        assert result.message == "Understood. Cancelled."
        assert engine.get_current_goal() is None

    def test_mixed_cancel_plus_status_does_not_cancel(self, engine):
        engine.set_goal(
            ActiveGoal(
                goal_id="goal_2",
                description="check weather",
                intent="weather:forecast",
                entities={},
            )
        )

        result = engine.resolve_goal("never mind, how are you today?", "status_inquiry", {})

        assert result.cancelled is False
        assert result.intent == "status_inquiry"
        assert result.message is None

    def test_polite_cancel_phrase_still_cancels(self, engine):
        engine.set_goal(
            ActiveGoal(
                goal_id="goal_3",
                description="send email",
                intent="email:send",
                entities={},
            )
        )

        result = engine.resolve_goal("cancel that please", "conversation:ack", {})

        assert result.cancelled is True
        assert result.message == "Understood. Cancelled."
        assert engine.get_current_goal() is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
