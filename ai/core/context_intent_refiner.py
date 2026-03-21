"""Context-aware intent refinement for ambiguous wording across domains."""

from __future__ import annotations

from typing import Dict, Optional


class ContextIntentRefiner:
    """Adjust intent label/confidence using recent conversational context."""

    def refine(
        self,
        *,
        user_input: str,
        intent: str,
        confidence: float,
        recent_topic: str = "",
        last_intent: str = "",
    ) -> Dict[str, object]:
        text = (user_input or "").lower()
        cur_intent = str(intent or "conversation:general")
        conf = max(0.0, min(1.0, float(confidence or 0.0)))
        topic = (recent_topic or "").lower()
        last = (last_intent or "").lower()

        # "analyze" after debugging context should stay technical.
        if "analyze" in text and any(k in (topic + " " + last) for k in ("debug", "traceback", "code", "bug")):
            if not cur_intent.startswith(("file_operations:", "notes:", "conversation:question")):
                cur_intent = "conversation:question"
            conf = max(conf, 0.74)
            return {"intent": cur_intent, "confidence": conf, "reason": "debug_context_refine"}

        # Financial context disambiguation
        if "analyze" in text and any(k in (topic + " " + last) for k in ("portfolio", "stocks", "invest", "market")):
            cur_intent = "conversation:question"
            conf = max(conf, 0.72)
            return {"intent": cur_intent, "confidence": conf, "reason": "finance_context_refine"}

        # Generic short follow-up inherits last actionable domain lightly.
        if len(text.split()) <= 6 and cur_intent.startswith("conversation:") and ":" in last and not last.startswith("conversation:"):
            cur_intent = last
            conf = max(conf, 0.67)
            return {"intent": cur_intent, "confidence": conf, "reason": "short_followup_domain_inherit"}

        return {"intent": cur_intent, "confidence": conf, "reason": "no_change"}
