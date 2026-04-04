"""Reference resolver for short follow-ups and pronouns."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

from ai.core.entity_registry import get_entity_registry


@dataclass
class ResolutionResult:
    rewritten_input: str
    resolved_bindings: Dict[str, str] = field(default_factory=dict)
    unresolved_pronouns: List[str] = field(default_factory=list)


class ReferenceResolver:
    PRONOUNS = ("it", "that", "this", "them", "those")
    _TEMPORAL_DEICTIC = {
        "week",
        "weekend",
        "month",
        "year",
        "morning",
        "afternoon",
        "evening",
        "night",
        "tonight",
        "today",
        "tomorrow",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "spring",
        "summer",
        "autumn",
        "fall",
        "winter",
    }

    def resolve(self, user_input: str, state: Dict[str, object]) -> ResolutionResult:
        text = (user_input or "").strip()
        if not text:
            return ResolutionResult(rewritten_input="")

        subject = self._pick_subject(state)
        if not subject:
            try:
                subject = str(
                    get_entity_registry().resolve_reference(text) or ""
                ).strip()
            except Exception:
                subject = ""
        resolved_bindings: Dict[str, str] = {}
        unresolved_pronouns: List[str] = []
        rewritten = text

        # Preserve local antecedents in the same turn, such as
        # "my ai ... it ...", instead of forcing cross-turn substitution.
        local_ai_antecedent = bool(
            re.search(r"\b(?:my|the|this|that)\s+ai\b", text, flags=re.IGNORECASE)
            and re.search(r"\bit\b", text, flags=re.IGNORECASE)
        )

        for pronoun in self.PRONOUNS:
            if local_ai_antecedent and pronoun == "it":
                continue
            if pronoun == "this" and self._is_temporal_deictic_usage(rewritten):
                continue
            if re.search(rf"\b{re.escape(pronoun)}\b", rewritten, flags=re.IGNORECASE):
                if subject:
                    rewritten = re.sub(
                        rf"\b{re.escape(pronoun)}\b",
                        subject,
                        rewritten,
                        count=1,
                        flags=re.IGNORECASE,
                    )
                    resolved_bindings[pronoun] = subject
                else:
                    unresolved_pronouns.append(pronoun)

        if subject:
            try:
                get_entity_registry().register(
                    label=subject,
                    entity_type="subject",
                    source="reference_resolver",
                    metadata={"resolved_bindings": dict(resolved_bindings or {})},
                )
            except Exception:
                pass

        return ResolutionResult(
            rewritten_input=rewritten,
            resolved_bindings=resolved_bindings,
            unresolved_pronouns=unresolved_pronouns,
        )

    @classmethod
    def _is_temporal_deictic_usage(cls, text: str) -> bool:
        match = re.search(r"\bthis\s+([a-z]+)\b", text or "", flags=re.IGNORECASE)
        if not match:
            return False
        return match.group(1).lower() in cls._TEMPORAL_DEICTIC

    @staticmethod
    def _pick_subject(state: Dict[str, object]) -> str:
        for key in ("last_subject", "current_topic", "active_goal"):
            val = state.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()

        refs = state.get("referenced_entities")
        if isinstance(refs, list) and refs:
            last = refs[-1]
            if isinstance(last, str) and last.strip():
                return last.strip()

        entities = state.get("last_entities")
        if isinstance(entities, dict):
            for key in ("resolved_reference", "file_path", "topic", "subject"):
                val = entities.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
        return ""
