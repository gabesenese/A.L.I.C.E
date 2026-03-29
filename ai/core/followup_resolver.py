"""Follow-up resolution logic for Foundation 2 routing."""

from dataclasses import dataclass
from typing import Dict, List, Optional
import math
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class FollowUpResult:
    """Result of a follow-up resolution attempt."""

    resolved_intent: str
    confidence: float
    was_followup: bool
    domain: Optional[str]
    reason: str


class FollowUpResolver:
    """Single authority for cross-turn follow-up inheritance."""

    DOMAIN_SIGNALS: Dict[str, List[str]] = {
        "weather": [
            "wear",
            "layer",
            "coat",
            "jacket",
            "bring",
            "umbrella",
            "need",
            "cold",
            "warm",
            "snow",
            "rain",
            "forecast",
            "tomorrow",
            "tonight",
            "this week",
            "next week",
            "weekend",
            "what about",
            "humidity",
            "wind",
            "feel like",
            "chilly",
            "freezing",
            "hot",
            "sunny",
            "cloudy",
            "dress",
        ],
        "notes": [
            "add to",
            "delete",
            "remove",
            "modify",
            "change",
            "show",
            "that note",
            "edit",
            "update",
            "what is in",
            "what's in",
            "inside",
            "read it",
            "open it",
            "what does it say",
            "what does it contain",
            "what's inside",
            "contents",
        ],
        "email": [
            "that email",
            "reply",
            "delete it",
            "archive",
            "the first one",
            "the latest",
            "forward",
            "respond",
        ],
        "music": [
            "that song",
            "skip",
            "pause",
            "volume",
            "louder",
            "quieter",
            "next track",
            "previous",
            "shuffle",
        ],
        "calendar": [
            "reschedule",
            "cancel",
            "that event",
            "the meeting",
            "move it",
            "postpone",
        ],
        "reminder": [
            "that reminder",
            "cancel it",
            "postpone",
            "snooze",
            "remind me again",
            "change the time",
        ],
    }

    CONVERSATIONAL_INTENTS = frozenset(
        {
            "greeting",
            "conversation:general",
            "conversation:question",
            "vague_question",
            "vague_temporal_question",
        }
    )

    _GENERIC_CUE_RE: re.Pattern = re.compile(
        "|".join(
            r"\\b" + re.escape(cue.strip()) + r"\\b"
            for cue in [
                "what about",
                "how about",
                "and",
                "also",
                "same for",
                "that",
                "this",
                "it",
                "them",
                "tomorrow",
                "tonight",
                "this week",
                "next week",
                "weekend",
            ]
        ),
        re.IGNORECASE,
    )

    def resolve(
        self,
        user_input: str,
        nlp_intent: str,
        nlp_confidence: float,
        last_intent: Optional[str],
        conversation_topics: Optional[List[str]] = None,
        perception_followup_domain: Optional[str] = None,
        turn_distance: int = 0,
    ) -> FollowUpResult:
        topics = conversation_topics or []
        recent_intent: Optional[str] = last_intent or (topics[-1] if topics else None)
        lower = user_input.lower().strip()

        if not recent_intent:
            return FollowUpResult(nlp_intent, nlp_confidence, False, None, "no_recent_context")

        recent_domain = recent_intent.split(":")[0] if ":" in recent_intent else recent_intent
        generic_cue = bool(self._GENERIC_CUE_RE.search(lower))

        domain_signals = self.DOMAIN_SIGNALS.get(recent_domain, [])
        domain_signal_hits = [sig for sig in domain_signals if sig in lower]
        domain_signal_hit = bool(domain_signal_hits)

        if recent_domain == "weather" and domain_signal_hit:
            weak_weather_signals = {"need", "bring"}
            strong_weather_hits = [sig for sig in domain_signal_hits if sig not in weak_weather_signals]
            short_generic_followup = generic_cue and len(lower.split()) <= 6
            if not strong_weather_hits and not short_generic_followup:
                domain_signal_hit = False

        is_conversational = nlp_intent in self.CONVERSATIONAL_INTENTS
        low_confidence = nlp_confidence < 0.7
        nlp_domain = nlp_intent.split(":")[0] if ":" in nlp_intent else nlp_intent
        same_domain_specific = (
            nlp_domain == recent_domain
            and nlp_intent not in self.CONVERSATIONAL_INTENTS
            and not nlp_intent.endswith(":general")
        )

        if domain_signal_hit and recent_domain in self.DOMAIN_SIGNALS and not same_domain_specific:
            decay = math.exp(-0.15 * max(0, turn_distance))
            new_conf = max(nlp_confidence, 0.82 * decay)

            resolved_intent = recent_intent
            if recent_domain == "notes":
                note_content_signals = frozenset(
                    {
                        "what is in",
                        "what's in",
                        "what is inside",
                        "what's inside",
                        "inside",
                        "read it",
                        "open it",
                        "what does it say",
                        "what does it contain",
                        "contents",
                        "in it",
                        "in the note",
                    }
                )
                note_delete_signals = frozenset({"delete", "remove"})
                note_append_signals = frozenset({"add to", "append"})
                note_edit_signals = frozenset({"edit", "modify", "change", "update"})

                if any(sig in lower for sig in note_content_signals):
                    resolved_intent = "notes:read_content"
                elif any(sig in lower for sig in note_delete_signals):
                    resolved_intent = "notes:delete"
                elif any(sig in lower for sig in note_append_signals):
                    resolved_intent = "notes:append"
                elif any(sig in lower for sig in note_edit_signals):
                    resolved_intent = "notes:edit"

            logger.debug(
                "[FollowUpResolver] domain_signal:%s -> %s (%.2f, decay=%.2f)",
                recent_domain,
                resolved_intent,
                new_conf,
                decay,
            )
            return FollowUpResult(
                resolved_intent,
                new_conf,
                True,
                recent_domain,
                f"domain_signal:{recent_domain}",
            )

        if perception_followup_domain and perception_followup_domain == recent_domain and not same_domain_specific:
            decay = math.exp(-0.15 * max(0, turn_distance))
            new_conf = max(nlp_confidence, 0.80 * decay)
            logger.debug(
                "[FollowUpResolver] perception_signal:%s -> %s (%.2f, decay=%.2f)",
                recent_domain,
                recent_intent,
                new_conf,
                decay,
            )
            return FollowUpResult(
                recent_intent,
                new_conf,
                True,
                recent_domain,
                f"perception_signal:{recent_domain}",
            )

        if low_confidence and (is_conversational or generic_cue):
            nlp_is_specific_pivot = (
                nlp_confidence >= 0.50
                and nlp_intent not in self.CONVERSATIONAL_INTENTS
                and not nlp_intent.endswith(":general")
                and nlp_domain != recent_domain
            )
            if not nlp_is_specific_pivot and ":" in recent_intent and not recent_intent.startswith(("conversation:", "system:")):
                decay = math.exp(-0.15 * max(0, turn_distance))
                new_conf = max(nlp_confidence, 0.78 * decay)
                logger.debug(
                    "[FollowUpResolver] generic_followup -> %s (%.2f, decay=%.2f)",
                    recent_intent,
                    new_conf,
                    decay,
                )
                return FollowUpResult(recent_intent, new_conf, True, recent_domain, "generic_followup")

        return FollowUpResult(nlp_intent, nlp_confidence, False, None, "no_followup")
