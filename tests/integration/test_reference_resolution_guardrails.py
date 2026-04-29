"""Integration tests for reference-resolution guardrails."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai.core.reasoning_engine import EntityKind, ReasoningEngine, WorldEntity


class TestReferenceResolutionGuardrails:
    def setup_method(self):
        self.engine = ReasoningEngine(user_name="Gabriel")

    def test_code_analysis_query_does_not_resolve_this_to_recent_weather_entity(self):
        self.engine.add_entity(
            WorldEntity(
                id="w_1",
                kind=EntityKind.TOPIC,
                label="Current weather in Kitchener",
            )
        )

        result = self.engine.resolve_references(
            "can you summarize what this file is doing?"
        )

        assert "this" not in result.bindings
        assert result.resolved_input == "can you summarize what this file is doing?"

    def test_regular_pronoun_resolution_still_works_for_action_queries(self):
        self.engine.add_entity(
            WorldEntity(
                id="n_1",
                kind=EntityKind.NOTE,
                label="sprint plan",
            )
        )
        self.engine.last_intent = "notes:create"

        result = self.engine.resolve_references("open it")

        assert "it" in result.bindings
        assert "sprint plan" in result.resolved_input
