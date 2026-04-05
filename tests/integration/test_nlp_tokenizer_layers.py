"""Integration tests for layered tokenizer pipeline in NLPProcessor."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai.core.nlp_processor import NLPProcessor, ParsedCommand


EXACT_LOG_PROMPT = "let's imagine how jarvis would be created with today's technology no fiction"
EXACT_TONY_PROMPT = "let's imagine how tony stark would have created Jarvis with todays technology, no fiction"
EXACT_CREATE_PROMPT = "how can i create an ai just like jarvis but with todays technology"


class TestLayeredTokenizer:
    def setup_method(self):
        self.nlp = NLPProcessor()
        self.nlp.reset_context()
        self.nlp.set_tokenizer_profile("default")

    def test_surface_segments_meta_and_utterance(self):
        debug = self.nlp.debug_tokenizer("/debug tokens; read the test note")
        segments = debug.get("segments", [])
        assert len(segments) >= 2
        assert segments[0]["kind"] == "meta_command"

    def test_rich_tokens_include_roles_and_flags(self):
        debug = self.nlp.debug_tokenizer("read the second note #work")
        tokens = debug.get("tokens", [])
        assert any(t.get("kind") == "ordinal" for t in tokens)
        assert any(t.get("role") == "action" and t.get("normalized") == "read" for t in tokens)
        assert any(t.get("kind") == "hashtag" for t in tokens)

    def test_parsed_command_extracts_note_read_title(self):
        debug = self.nlp.debug_tokenizer("read the project alpha note for me")
        parsed = debug.get("parsed_command", {})
        assert parsed.get("action") == "read"
        assert parsed.get("object_type") == "note"
        assert "project alpha" in (parsed.get("title_hint") or "")

    def test_process_uses_structured_parse_for_notes_intent(self):
        result = self.nlp.process("do i have a note called test")
        assert result.intent.startswith("notes:")
        assert result.parsed_command.get("object_type") == "note"
        assert "notes" in result.plugin_scores

    def test_strict_profile_reduces_fuzzy_routing(self):
        self.nlp.set_tokenizer_profile("strict")
        debug = self.nlp.debug_tokenizer("please maybe check something quickly")
        scores = debug.get("plugin_scores", {})
        assert scores.get("conversation", 0) >= 0.2

    def test_noisy_channel_normalization_corrects_command_typos(self):
        debug = self.nlp.debug_tokenizer("raed the nots")
        normalized = debug.get("normalized_text", "")
        assert "read" in normalized
        assert "notes" in normalized

    def test_retrieval_first_short_followup_uses_context(self):
        self.nlp.process("create a note called sprint plan")
        result = self.nlp.process("read it")
        assert result.intent == "notes:read"
        assert result.intent_confidence >= 0.8

    def test_ambiguous_query_emits_disambiguation_metadata(self):
        result = self.nlp.process("do that thing")
        modifiers = result.parsed_command.get("modifiers", {})
        disambiguation = modifiers.get("disambiguation", {})
        assert isinstance(disambiguation, dict)
        assert disambiguation.get("needs_clarification") is True

    def test_process_exposes_top_three_intent_candidates(self):
        result = self.nlp.process("show my latest emails")
        assert isinstance(result.intent_candidates, list)
        assert 1 <= len(result.intent_candidates) <= 3
        assert all("intent" in item and "score" in item for item in result.intent_candidates)

    def test_weather_misroute_gets_unknown_fallback(self):
        result = self.nlp.process("can we brainstorm a plan for my week")
        assert result.intent == "conversation:clarification_needed"
        assert result.intent_plausibility < 0.75
        modifiers = result.parsed_command.get("modifiers", {})
        assert modifiers.get("unknown_intent_fallback") is True

    def test_category_gate_disables_tools_for_conversation_query(self):
        result = self.nlp.process("let's brainstorm architecture ideas for this project")
        modifiers = result.parsed_command.get("modifiers", {})
        assert modifiers.get("intent_category") == "conversation"
        assert modifiers.get("tool_execution_disabled") is True
        assert result.intent.startswith("conversation:")

    def test_goal_statement_is_classified_as_conversation_goal_statement(self):
        result = self.nlp.process(
            "I want to make Alice more autonomous and less like a chatbot"
        )
        assert result.intent == "conversation:goal_statement"
        assert result.intent_confidence >= 0.8

    def test_goal_statement_objective_phrase_is_recognized(self):
        result = self.nlp.process(
            "My objective is to build a reliable agent architecture with planning and verification"
        )
        assert result.intent == "conversation:goal_statement"
        assert result.intent_confidence >= 0.8

    def test_answerability_gate_routes_specific_domain_question_to_learning_explanation_request(self):
        result = self.nlp.process("how does optimizer training work in nlp models?")
        modifiers = result.parsed_command.get("modifiers", {})

        assert result.intent == "learning:explanation_request"
        assert modifiers.get("allow_direct_answer") is True
        assert modifiers.get("block_clarification") is True

    def test_answerability_gate_does_not_fire_for_ambiguous_domain_question(self):
        result = self.nlp.process("how does optimizer stuff work?")

        assert result.intent != "learning:explanation_request"

    def test_weather_plausibility_requires_entity_or_forecast_signals(self):
        score, issues = self.nlp._validate_intent_plausibility(
            "please help with this", "weather:current", ParsedCommand(), {}
        )
        assert score < 0.45
        assert "missing_weather_required_entities" in issues

    def test_weather_negative_evidence_scoring_detects_contradictions(self):
        score, issues = self.nlp._validate_intent_plausibility(
            "can we brainstorm architecture and api design options",
            "weather:forecast",
            ParsedCommand(),
            {},
        )
        assert score < 0.35
        assert "negative_evidence_weather_contradiction" in issues

    def test_weather_signal_query_prefers_weather_over_notes_scores(self):
        debug = self.nlp.debug_tokenizer("is it gonna snow by any chance?")
        scores = debug.get("plugin_scores", {})

        assert scores.get("weather", 0.0) > scores.get("notes", 0.0)
        assert scores.get("weather", 0.0) >= 2.0

    def test_weather_signal_query_not_misclassified_as_notes(self):
        result = self.nlp.process("is it gonna snow by any chance?")

        assert result.intent.startswith("weather:")
        assert not result.intent.startswith("notes:")

    def test_memory_recall_phrase_not_misclassified_as_store(self):
        result = self.nlp.process("do you remember that coding problem we were talking about?")
        assert result.intent == "memory:recall"

    def test_notes_plausibility_penalizes_internal_code_query(self):
        score, issues = self.nlp._validate_intent_plausibility(
            "list your internal code?", "notes:list", ParsedCommand(), {}
        )
        assert score < 0.30
        assert "negative_evidence_notes_contradiction" in issues

    def test_followup_chain_benchmark_notes_selection_and_content(self):
        """Benchmark-style chain: first/second selection should stay in notes domain."""
        self.nlp.process("show my notes")
        first_pick = self.nlp.process("open the first one")
        second_pick = self.nlp.process("actually the second one")
        content = self.nlp.process("what is in it?")

        assert first_pick.intent.startswith("notes:")
        assert second_pick.intent.startswith("notes:")
        assert content.intent.startswith("notes:")

    def test_followup_chain_benchmark_cross_domain_pivot_is_respected(self):
        """Explicit pivot in a chain should not be dragged back by follow-up inheritance."""
        self.nlp.process("schedule a meeting tomorrow at 3pm")
        self.nlp.process("change that to tomorrow")
        pivot = self.nlp.process("send it to her too")

        assert not pivot.intent.startswith("calendar:")

    def test_followup_chain_benchmark_rejection_phrase_avoids_weather_bleed(self):
        """"not that one" after notes context should not misroute into weather."""
        self.nlp.process("show my notes")
        self.nlp.process("open the first one")
        rejection = self.nlp.process("not that one")

        assert not rejection.intent.startswith("weather:")

    def test_pending_help_narrowing_slot_consumes_short_answer(self):
        self.nlp.context.pending_clarification = {
            "type": "help_narrowing",
            "slot": "project_subdomain",
            "parent_topic": "ai_project",
            "expected_answer_shape": "short_topic_or_subdomain",
        }

        result = self.nlp.process("NLP")
        modifiers = result.parsed_command.get("modifiers", {})
        slot_followup = modifiers.get("pending_slot_followup", {})
        followup_resolution = modifiers.get("followup_resolution", {})

        assert result.intent == "conversation:clarification_needed"
        assert slot_followup.get("filled") is True
        assert slot_followup.get("slot") == "project_subdomain"
        assert slot_followup.get("value", "").lower() == "nlp"
        assert followup_resolution.get("was_followup") is True

    def test_pending_help_narrowing_slot_does_not_consume_help_opener(self):
        self.nlp.context.pending_clarification = {
            "type": "help_narrowing",
            "slot": "project_subdomain",
            "parent_topic": "ai_project",
            "expected_answer_shape": "short_topic_or_subdomain",
        }

        result = self.nlp.process("can you help me")
        modifiers = result.parsed_command.get("modifiers", {})

        assert modifiers.get("pending_slot_followup") is None

    def test_pending_help_narrowing_slot_marks_selected_reference_for_ordinals(self):
        self.nlp.context.pending_clarification = {
            "type": "help_narrowing",
            "slot": "project_subdomain",
            "parent_topic": "ai_project",
            "expected_answer_shape": "ordinal_or_short_phrase",
        }

        result = self.nlp.process("the second one")
        modifiers = result.parsed_command.get("modifiers", {})
        slot_followup = modifiers.get("pending_slot_followup", {})

        assert slot_followup.get("filled") is True
        assert slot_followup.get("selected_reference") == "second"
        assert modifiers.get("selected_object_reference") == "second"

    def test_pending_route_choice_slot_consumes_explanation(self):
        self.nlp.context.pending_clarification = {
            "type": "route_choice",
            "slot": "route_choice",
            "parent_topic": "conversation",
            "expected_answer_shape": "single_token",
        }

        result = self.nlp.process("explanation")
        modifiers = result.parsed_command.get("modifiers", {})
        slot_followup = modifiers.get("pending_slot_followup", {})

        assert result.intent == "conversation:clarification_needed"
        assert slot_followup.get("filled") is True
        assert slot_followup.get("slot") == "route_choice"
        assert slot_followup.get("value") == "explanation"

    def test_pending_route_choice_slot_persists_when_not_filled(self):
        self.nlp.context.pending_clarification = {
            "type": "route_choice",
            "slot_type": "route_choice",
            "slot": "route_choice",
            "parent_topic": "conversation",
            "expected_answer_shape": "single_token",
            "active": True,
        }

        result = self.nlp.process("okay")
        modifiers = result.parsed_command.get("modifiers", {})

        assert modifiers.get("pending_slot_followup") is None
        assert isinstance(self.nlp.context.pending_clarification, dict)
        assert self.nlp.context.pending_clarification.get("slot_type") == "route_choice"

    def test_pending_route_choice_slot_clears_after_fill(self):
        self.nlp.context.pending_clarification = {
            "type": "route_choice",
            "slot_type": "route_choice",
            "slot": "route_choice",
            "parent_topic": "conversation",
            "expected_answer_shape": "single_token",
            "active": True,
        }

        result = self.nlp.process("quick search")
        modifiers = result.parsed_command.get("modifiers", {})

        assert modifiers.get("pending_slot_followup", {}).get("filled") is True
        pending = self.nlp.context.pending_clarification
        pending_type = str((pending or {}).get("slot_type") or (pending or {}).get("type") or "").lower()
        assert pending_type != "route_choice"

    def test_rich_conceptual_prompt_not_classified_as_clarification_needed(self):
        result = self.nlp.process(EXACT_LOG_PROMPT)

        assert result.intent != "conversation:clarification_needed"
        assert result.intent in {"learning:system_design", "conversation:question"}
        assert result.intent_plausibility >= 0.70

    def test_route_choice_pending_slot_does_not_hijack_rich_conceptual_prompt(self):
        self.nlp.context.pending_clarification = {
            "type": "route_choice",
            "slot_type": "route_choice",
            "slot": "route_choice",
            "parent_topic": "conversation",
            "expected_answer_shape": "single_token",
            "active": True,
        }

        result = self.nlp.process(EXACT_LOG_PROMPT)
        modifiers = result.parsed_command.get("modifiers", {})

        assert result.intent != "conversation:clarification_needed"
        assert modifiers.get("pending_slot_followup") is None
        assert modifiers.get("unknown_intent_fallback") is False

    def test_exact_tony_prompt_routes_without_clarification_or_unknown_fallback(self):
        result = self.nlp.process(EXACT_TONY_PROMPT)
        modifiers = result.parsed_command.get("modifiers", {})

        assert result.intent != "conversation:clarification_needed"
        assert result.intent == "learning:system_design"
        assert modifiers.get("unknown_intent_fallback") is False

    def test_exact_create_prompt_routes_without_clarification_or_unknown_fallback(self):
        result = self.nlp.process(EXACT_CREATE_PROMPT)
        modifiers = result.parsed_command.get("modifiers", {})

        assert result.intent != "conversation:clarification_needed"
        assert result.intent == "learning:system_design"
        assert modifiers.get("unknown_intent_fallback") is False
