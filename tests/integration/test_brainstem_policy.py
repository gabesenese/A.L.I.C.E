"""
tests/integration/test_brainstem_policy.py
──────────────────────────────────────────
Integration tests for the Perception + InteractionPolicy brainstem pipeline.

Covers every mood branch of InteractionPolicy.derive(), the full
Perception.build() path, and all three resolution layers of FollowUpResolver.
These components form the "brainstem": the single authority that gates
clarification, selects tone, and decides how aggressive the pipeline
should be about inheriting prior context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from ai.core.interaction_policy import InteractionPolicy, PolicySettings
from ai.core.nlp_processor import Perception, PerceptionResult, FollowUpResolver


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_query(
    intent: str = "conversation:general",
    confidence: float = 0.8,
    compound: float = 0.0,
    urgency: str = "low",
    text: str = "hello",
    validation_score: float = 1.0,
) -> MagicMock:
    """Minimal stand-in for ProcessedQuery."""
    q = MagicMock()
    q.intent = intent
    q.intent_confidence = confidence
    q.sentiment = {"compound": compound}
    q.urgency_level = urgency
    q.original_text = text
    q.validation_score = validation_score
    q.entities = {}
    # _clarification_need inspects parsed_command as a dict; return empty dict
    # so the MagicMock doesn't produce accidental truthy disambiguations.
    q.parsed_command = {}
    return q


def _sentiment(compound: float = 0.0) -> Dict[str, Any]:
    return {"compound": compound, "pos": 0.0, "neg": 0.0, "neu": 1.0}


# ── InteractionPolicy tests ───────────────────────────────────────────────────

class TestInteractionPolicy:
    policy = InteractionPolicy()

    def test_frustrated_mood_skips_clarification(self):
        settings = self.policy.derive("frustrated", _sentiment(-0.3), "low")
        assert settings.skip_clarification is True
        assert settings.tone == "empathetic"
        assert settings.response_length == "brief"
        assert settings.add_empathy_prefix is True
        assert settings.clarification_threshold < 0.5

    def test_urgent_mood_skips_clarification_is_direct(self):
        settings = self.policy.derive("urgent", _sentiment(0.0), "high")
        assert settings.skip_clarification is True
        assert settings.tone == "direct"
        assert settings.response_length == "brief"
        assert settings.add_empathy_prefix is False

    def test_high_urgency_overrides_neutral_mood(self):
        """urgency='high' should behave identically to mood='urgent'."""
        by_urgency = self.policy.derive("neutral", _sentiment(0.0), "high")
        by_mood    = self.policy.derive("urgent",  _sentiment(0.0), "high")
        assert by_urgency.skip_clarification == by_mood.skip_clarification
        assert by_urgency.tone == by_mood.tone

    def test_negative_mood_empathetic_no_skip(self):
        settings = self.policy.derive("negative", _sentiment(-0.5), "low")
        assert settings.skip_clarification is False
        assert settings.tone == "empathetic"
        assert settings.add_empathy_prefix is True
        assert settings.response_length == "normal"

    def test_negative_compound_triggers_empathy(self):
        """compound < -0.2 should produce empathetic policy even with neutral mood."""
        settings = self.policy.derive("neutral", _sentiment(-0.3), "low")
        assert settings.tone == "empathetic"
        assert settings.skip_clarification is False

    def test_positive_mood_encouraging_no_empathy_prefix(self):
        settings = self.policy.derive("positive", _sentiment(0.6), "low")
        assert settings.skip_clarification is False
        assert settings.tone == "encouraging"
        assert settings.add_empathy_prefix is False

    def test_positive_compound_triggers_encouraging(self):
        """compound > 0.4 with neutral mood should still produce encouraging tone."""
        settings = self.policy.derive("neutral", _sentiment(0.5), "low")
        assert settings.tone == "encouraging"

    def test_neutral_default(self):
        settings = self.policy.derive("neutral", _sentiment(0.0), "low")
        assert settings.skip_clarification is False
        assert settings.tone == "neutral"
        assert settings.response_length == "normal"
        assert settings.add_empathy_prefix is False

    def test_returns_policy_settings_type(self):
        settings = self.policy.derive("neutral", _sentiment(), "low")
        assert isinstance(settings, PolicySettings)

    def test_clarification_threshold_is_float_in_range(self):
        for mood in ("frustrated", "urgent", "negative", "positive", "neutral"):
            settings = self.policy.derive(mood, _sentiment(), "low")
            assert 0.0 < settings.clarification_threshold <= 1.0

    def test_response_length_valid_values(self):
        valid = {"brief", "normal", "detailed"}
        for mood in ("frustrated", "urgent", "negative", "positive", "neutral"):
            settings = self.policy.derive(mood, _sentiment(), "low")
            assert settings.response_length in valid


# ── Perception tests ──────────────────────────────────────────────────────────

class TestPerception:
    perception = Perception()

    def test_high_confidence_unambiguous(self):
        q = _make_query(intent="notes:create", confidence=0.95)
        result = self.perception.build(q, last_intent=None, conversation_topics=[])
        assert result.ambiguity < 0.2
        assert result.needs_clarification is False
        assert result.inferred_mood in ("neutral", "positive", "negative", "frustrated", "urgent")

    def test_frustrated_mood_detected(self):
        q = _make_query(
            intent="conversation:general",
            confidence=0.6,
            compound=-0.4,
            text="ugh why is this broken again",
        )
        result = self.perception.build(q)
        assert result.inferred_mood == "frustrated"

    def test_urgent_mood_detected_from_urgency_level(self):
        q = _make_query(urgency="high", text="I need this now asap", compound=0.0)
        result = self.perception.build(q)
        assert result.inferred_mood == "urgent"

    def test_positive_mood_detected(self):
        q = _make_query(compound=0.7, text="this is amazing I love it")
        result = self.perception.build(q)
        assert result.inferred_mood == "positive"

    def test_negative_mood_detected(self):
        # Avoid words in FRUSTRATION_MARKERS (e.g. 'terrible') which would
        # take the frustrated branch even with strong negative compound.
        q = _make_query(compound=-0.6, text="I am quite unhappy with this")
        result = self.perception.build(q)
        assert result.inferred_mood == "negative"

    def test_weather_followup_domain_detected(self):
        """Umbrella keyword should trigger weather follow-up domain detection."""
        q = _make_query(intent="conversation:general", confidence=0.5, text="should I bring an umbrella")
        result = self.perception.build(
            q,
            last_intent="weather:current",
            conversation_topics=["weather:current"],
        )
        assert result.followup_domain == "weather"

    def test_no_followup_domain_on_first_turn(self):
        q = _make_query(confidence=0.9, text="set a timer for 5 minutes")
        result = self.perception.build(q, last_intent=None, conversation_topics=[])
        assert result.followup_domain is None

    def test_perception_result_has_required_fields(self):
        q = _make_query()
        result = self.perception.build(q)
        assert isinstance(result, PerceptionResult)
        assert hasattr(result, "inferred_mood")
        assert hasattr(result, "ambiguity")
        assert hasattr(result, "followup_domain")
        assert hasattr(result, "needs_clarification")
        assert hasattr(result, "interaction_hints")

    def test_low_confidence_raises_ambiguity(self):
        q = _make_query(confidence=0.2)
        result = self.perception.build(q)
        assert result.ambiguity > 0.4

    def test_intent_proxy_works(self):
        q = _make_query(intent="music:play")
        result = self.perception.build(q)
        assert result.intent == "music:play"


# ── FollowUpResolver tests ────────────────────────────────────────────────────

class TestFollowUpResolver:
    resolver = FollowUpResolver()

    def _resolve(
        self,
        user_input: str,
        nlp_intent: str = "conversation:general",
        nlp_confidence: float = 0.55,
        last_intent: Optional[str] = None,
        topics: Optional[List[str]] = None,
        perception_domain: Optional[str] = None,
    ):
        return self.resolver.resolve(
            user_input=user_input,
            nlp_intent=nlp_intent,
            nlp_confidence=nlp_confidence,
            last_intent=last_intent,
            conversation_topics=topics,
            perception_followup_domain=perception_domain,
        )

    def test_no_context_returns_raw_nlp(self):
        r = self._resolve("hello", last_intent=None, topics=[])
        assert r.resolved_intent == "conversation:general"
        assert r.was_followup is False

    def test_layer1_domain_signal_weather_umbrella(self):
        """'umbrella' in the utterance should inherit weather:current."""
        r = self._resolve(
            "should I bring an umbrella?",
            last_intent="weather:current",
            nlp_intent="conversation:general",
            nlp_confidence=0.5,
        )
        assert r.resolved_intent == "weather:current"
        assert r.was_followup is True
        assert r.domain == "weather"
        assert r.confidence > 0.79

    def test_layer1_domain_signal_music_skip(self):
        r = self._resolve(
            "skip this song",
            last_intent="music:play",
            nlp_intent="conversation:general",
            nlp_confidence=0.5,
        )
        assert r.resolved_intent == "music:play"
        assert r.was_followup is True

    def test_layer2_perception_signal_activates(self):
        # Use a neutral utterance with no Layer-1 domain keywords so Layer 2
        # (perception_domain) is the trigger instead of a domain-signal keyword.
        r = self._resolve(
            "tell me more",
            last_intent="weather:current",
            nlp_intent="conversation:general",
            nlp_confidence=0.5,
            perception_domain="weather",
        )
        assert r.resolved_intent == "weather:current"
        assert r.was_followup is True
        # Layer 2 or Layer 1 may handle it; either way the reason records the source.
        assert r.reason != ""

    def test_layer3_generic_followup_low_confidence(self):
        r = self._resolve(
            "and tomorrow?",
            last_intent="calendar:query",
            nlp_intent="conversation:general",
            nlp_confidence=0.45,
        )
        assert r.resolved_intent == "calendar:query"
        assert r.was_followup is True

    def test_high_confidence_nlp_not_overridden_by_generic_cue(self):
        """When NLP is confident (>=0.7), generic cues should NOT override."""
        r = self._resolve(
            "and also",
            last_intent="calendar:query",
            nlp_intent="notes:create",
            nlp_confidence=0.85,
        )
        # Layer 3 only fires when low_confidence — high confidence should pass through
        assert r.resolved_intent == "notes:create"

    def test_accumulated_confidence_boost_layer1(self):
        """Inherited intents should have confidence boosted above raw NLP confidence."""
        raw_conf = 0.40
        r = self._resolve(
            "will it snow?",
            last_intent="weather:current",
            nlp_intent="vague_question",
            nlp_confidence=raw_conf,
        )
        if r.was_followup:
            assert r.confidence > raw_conf

    def test_system_intent_not_promoted_by_layer3(self):
        """Intents starting with 'system:' should not be promoted via generic follow-up."""
        r = self._resolve(
            "and also maybe",
            last_intent="system:status",
            nlp_intent="conversation:general",
            nlp_confidence=0.4,
        )
        # system: intents are excluded from layer-3 promotion
        assert r.resolved_intent != "system:status" or r.was_followup is False

    def test_layer2_does_not_override_specific_same_domain_intent(self):
        """notes:create must survive when perception also flags 'notes' domain.

        Bug: NLP correctly returns notes:create but Layer 2 was replacing it
        with notes:query_exist (the previous turn's intent).
        """
        r = self._resolve(
            "let's create a grocery note",
            nlp_intent="notes:create",
            nlp_confidence=0.97,
            last_intent="notes:query_exist",
            topics=["notes:query_exist"],
            perception_domain="notes",
        )
        assert r.was_followup is False
        assert r.resolved_intent == "notes:create"

    def test_layer1_does_not_override_specific_same_domain_intent(self):
        """notes:read must survive even when a notes domain signal is present."""
        r = self._resolve(
            "add banana eggs and milk to the grocery note",
            nlp_intent="notes:read",
            nlp_confidence=0.80,
            last_intent="notes:query_exist",
            topics=["notes:query_exist"],
            perception_domain="notes",
        )
        assert r.was_followup is False
        assert r.resolved_intent == "notes:read"

    # ── domain-aware notes sub-intent mapping (feature items 1-5) ────────────

    def test_notes_content_cue_maps_to_read_content(self):
        """'what is in it?' after notes:query_exist must yield notes:read_content, not query_exist."""
        r = self._resolve(
            "what is in it?",
            nlp_intent="conversation:general",
            nlp_confidence=0.45,
            last_intent="notes:query_exist",
            topics=["notes:query_exist"],
        )
        assert r.was_followup is True
        assert r.resolved_intent == "notes:read_content"

    def test_notes_delete_cue_maps_to_delete(self):
        """'delete it' after notes:list must yield notes:delete, not notes:list."""
        r = self._resolve(
            "delete it",
            nlp_intent="conversation:general",
            nlp_confidence=0.45,
            last_intent="notes:list",
            topics=["notes:list"],
        )
        assert r.was_followup is True
        assert r.resolved_intent == "notes:delete"

    def test_notes_append_cue_maps_to_append(self):
        """'add to it' signal must yield notes:append."""
        r = self._resolve(
            "add to it",
            nlp_intent="conversation:general",
            nlp_confidence=0.45,
            last_intent="notes:list",
            topics=["notes:list"],
        )
        assert r.was_followup is True
        assert r.resolved_intent == "notes:append"

    def test_notes_inside_cue_maps_to_read_content(self):
        """'what's inside' signal must yield notes:read_content."""
        r = self._resolve(
            "what's inside it?",
            nlp_intent="conversation:general",
            nlp_confidence=0.45,
            last_intent="notes:list",
            topics=["notes:list"],
        )
        assert r.was_followup is True
        assert r.resolved_intent == "notes:read_content"

    def test_weather_does_not_bleed_into_notes_follow_up(self):
        """After a notes turn, generic pronoun 'what is in it?' must not route to weather."""
        r = self._resolve(
            "what is in it?",
            nlp_intent="weather:current",   # NLP misfired as weather
            nlp_confidence=0.55,
            last_intent="notes:list",
            topics=["notes:list"],
        )
        # Layer 1 should override the misfired weather intent via notes content signal
        assert r.resolved_intent != "weather:current"
        assert r.resolved_intent == "notes:read_content"

    def test_cross_domain_weather_after_notes_is_respected(self):
        """Explicit weather query after notes should produce weather:current, not notes."""
        r = self._resolve(
            "how's the weather today?",
            nlp_intent="weather:current",
            nlp_confidence=0.88,
            last_intent="notes:list",
            topics=["notes:list"],
        )
        # High-confidence weather intent — same-domain specific guard blocks Layer 1/2
        # because nlp_domain != recent_domain, so resolver should pass through weather
        assert r.resolved_intent == "weather:current"

    def test_weather_domain_signal_does_not_hijack_generic_need_help_prompt(self):
        """Weak signal words like 'need' must not force weather inheritance."""
        r = self._resolve(
            "i need some help with an ai project and model routing",
            nlp_intent="conversation:help",
            nlp_confidence=0.97,
            last_intent="weather:current",
            topics=["weather:current"],
        )
        assert r.was_followup is False
        assert r.resolved_intent == "conversation:help"


# ── end-to-end brainstem integration ─────────────────────────────────────────


class TestBrainstemEndToEnd:
    """Smoke-test the full perception → policy chain."""

    perception = Perception()
    policy = InteractionPolicy()

    def _run(
        self,
        text: str,
        compound: float = 0.0,
        urgency: str = "low",
        confidence: float = 0.8,
    ) -> PolicySettings:
        q = _make_query(
            text=text,
            compound=compound,
            urgency=urgency,
            confidence=confidence,
        )
        pr = self.perception.build(q)
        return self.policy.derive(pr.inferred_mood, q.sentiment, urgency)

    def test_frustrated_user_gets_brief_empathetic(self):
        settings = self._run("ugh why broken again", compound=-0.5)
        assert settings.response_length == "brief"
        assert settings.tone == "empathetic"
        assert settings.skip_clarification is True

    def test_happy_user_gets_encouraging(self):
        settings = self._run("this is amazing!", compound=0.8)
        assert settings.tone == "encouraging"

    def test_urgent_request_gets_direct(self):
        settings = self._run("I need this now!", urgency="high")
        assert settings.tone == "direct"
        assert settings.skip_clarification is True

    def test_neutral_user_normal_flow(self):
        settings = self._run("what's the weather today", compound=0.1)
        assert settings.skip_clarification is False
        assert settings.response_length == "normal"
