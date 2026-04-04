"""Resolve clarification replies into executable follow-up actions."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import re
from typing import Any, Dict


@dataclass
class ClarificationResolution:
    consumed: bool = False
    route_choice: str = ""
    selected_branch: str = ""
    slot_type: str = ""
    reconstructed_input: str = ""
    reconstructed_intent: str = ""
    reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ClarificationResolver:
    """Consume pending clarification slots before generic routing."""

    def resolve(self, user_input: str, pending_slot: Dict[str, Any] | None) -> ClarificationResolution:
        slot = dict(pending_slot or {})
        slot_type = str(slot.get("slot_type") or slot.get("type") or "").strip().lower()
        text = str(user_input or "").strip()
        text_low = text.lower()

        if not text or not slot_type:
            return ClarificationResolution()

        if slot_type in {"route_choice", "help_route_choice"}:
            choice = self._parse_route_choice(text_low)
            if not choice:
                return ClarificationResolution(consumed=False, slot_type=slot_type)

            parent_request = str(
                slot.get("parent_request")
                or slot.get("parent_query")
                or slot.get("source_user_input")
                or ""
            ).strip()
            parent_intent = str(slot.get("parent_intent") or "conversation:help").strip()

            reconstructed_input = parent_request
            reconstructed_intent = parent_intent
            if choice == "quick_search":
                reconstructed_intent = "search"
                if not reconstructed_input:
                    reconstructed_input = str(slot.get("parent_topic") or "").strip()

            return ClarificationResolution(
                consumed=True,
                route_choice=choice,
                slot_type=slot_type,
                reconstructed_input=reconstructed_input,
                reconstructed_intent=reconstructed_intent,
                reason="route_choice_slot_consumed",
            )

        if slot_type in {"topic_branch", "help_topic_branch", "project_branch"}:
            branch = self._parse_topic_branch(
                text_low,
                allowed_values=list(slot.get("allowed_values") or []),
            )
            if not branch:
                return ClarificationResolution(consumed=False, slot_type=slot_type)

            parent_request = str(slot.get("parent_request") or "").strip()
            parent_topic = str(slot.get("parent_topic") or "topic").strip()
            parent_intent = str(slot.get("parent_intent") or "conversation:question").strip()

            if parent_request:
                reconstructed_input = f"{parent_request}. Focus first on {branch.replace('_', ' ')}."
            else:
                reconstructed_input = f"Give me a practical overview of {parent_topic} with focus on {branch.replace('_', ' ')}."

            return ClarificationResolution(
                consumed=True,
                selected_branch=branch,
                slot_type=slot_type,
                reconstructed_input=reconstructed_input,
                reconstructed_intent=parent_intent,
                reason="topic_branch_slot_consumed",
            )

        return ClarificationResolution()

    @staticmethod
    def _parse_route_choice(text_low: str) -> str:
        if re.search(r"\b(explanation|explain|walk\s*through|teach|how|why)\b", text_low):
            return "explanation"
        if re.search(r"\b(direct\s+action|action|do\s+it|execute|perform|run\s+it)\b", text_low):
            return "direct_action"
        if re.search(r"\b(quick\s+search|search|look\s*up|find\s+info|web|online)\b", text_low):
            return "quick_search"
        return ""

    @staticmethod
    def _parse_topic_branch(text_low: str, allowed_values: list[str]) -> str:
        value_map = {
            "intent_routing": (r"\b(intent\s*routing|routing|router)\b",),
            "entity_extraction": (r"\b(entity\s*extraction|entities|slot\s*filling|ner)\b",),
            "embeddings": (r"\b(embedding|embeddings|vector|sentence\s*embedding)\b",),
            "conversation_flow": (r"\b(conversation\s*flow|dialogue\s*flow|dialog\s*flow|flow)\b",),
        }

        normalized_allowed = [str(v).strip().lower() for v in list(allowed_values or []) if str(v).strip()]
        candidates = normalized_allowed or list(value_map.keys())
        for candidate in candidates:
            patterns = value_map.get(candidate, ())
            for pat in patterns:
                if re.search(pat, text_low):
                    return candidate

        if re.search(r"\b(first|1st|one|start with)\b", text_low) and candidates:
            return candidates[0]
        if re.search(r"\b(second|2nd|two)\b", text_low) and len(candidates) > 1:
            return candidates[1]
        if re.search(r"\b(third|3rd|three)\b", text_low) and len(candidates) > 2:
            return candidates[2]
        if re.search(r"\b(fourth|4th|four)\b", text_low) and len(candidates) > 3:
            return candidates[3]

        return ""


_clarification_resolver: ClarificationResolver | None = None


def get_clarification_resolver() -> ClarificationResolver:
    global _clarification_resolver
    if _clarification_resolver is None:
        _clarification_resolver = ClarificationResolver()
    return _clarification_resolver
