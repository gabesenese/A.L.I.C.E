"""Pre-NLP context resolver stage.

Pipeline target:
INPUT -> context resolver -> NLP classify -> planning -> tools
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Dict, List

from ai.ambiguity_detector import AmbiguityDetector
from ai.reference_resolver import ReferenceResolver


@dataclass
class ContextResolution:
    original_input: str
    rewritten_input: str
    rewrite_confidence: float = 0.0
    resolved_bindings: Dict[str, str] = field(default_factory=dict)
    needs_clarification: bool = False
    unresolved_pronouns: List[str] = field(default_factory=list)
    clarification_options: List[str] = field(default_factory=list)


class ContextResolver:
    """Resolves references before NLP classification."""

    def __init__(self) -> None:
        self.reference_resolver = ReferenceResolver()
        self.ambiguity_detector = AmbiguityDetector()
        # Rewrites must be extremely high-confidence; raw user text is authoritative.
        self.min_rewrite_confidence = 0.96

    @staticmethod
    def _contains_placeholder_noise(text: str) -> bool:
        low = str(text or "").lower()
        if "general_assistance" in low:
            return True
        if re.search(r"\b(?:person|assistant|system)\s+'[^']+'", low):
            return True
        if re.search(r"\bperson\s+'an ai'\b", low):
            return True
        if re.search(r"\b(?:unknown|placeholder|entity\s+'[^']+')\b", low):
            return True
        return False

    def _estimate_rewrite_confidence(
        self,
        *,
        original: str,
        rewritten: str,
        bindings: Dict[str, str],
    ) -> float:
        if not rewritten or rewritten == original:
            return 1.0
        if not bindings:
            return 0.0
        if self._contains_placeholder_noise(rewritten):
            return 0.0

        orig_tokens = re.findall(r"\b[a-z0-9']+\b", (original or "").lower())
        rew_tokens = re.findall(r"\b[a-z0-9']+\b", (rewritten or "").lower())
        if not orig_tokens or not rew_tokens:
            return 0.0

        overlap = len(set(orig_tokens).intersection(rew_tokens)) / max(1, len(set(orig_tokens)))
        length_delta = abs(len(rewritten) - len(original))
        length_penalty = 0.0 if length_delta <= 24 else min(0.40, length_delta / 200.0)
        binding_bonus = min(0.20, 0.05 * len(bindings))
        confidence = max(0.0, min(1.0, 0.75 + (0.25 * overlap) + binding_bonus - length_penalty))
        return confidence

    def resolve(self, user_input: str, state: Dict[str, object]) -> ContextResolution:
        result = self.reference_resolver.resolve(user_input=user_input, state=state)
        token_count = len((user_input or "").split())
        needs_clarification = self.ambiguity_detector.should_clarify(
            unresolved_pronouns=result.unresolved_pronouns,
            token_count=token_count,
        )

        rewritten_input = result.rewritten_input or user_input
        rewrite_confidence = self._estimate_rewrite_confidence(
            original=user_input,
            rewritten=rewritten_input,
            bindings=result.resolved_bindings,
        )
        if rewrite_confidence < self.min_rewrite_confidence:
            rewritten_input = user_input
        if self._contains_placeholder_noise(rewritten_input):
            rewritten_input = user_input
            rewrite_confidence = 0.0

        return ContextResolution(
            original_input=user_input,
            rewritten_input=rewritten_input,
            rewrite_confidence=rewrite_confidence,
            resolved_bindings=result.resolved_bindings,
            needs_clarification=needs_clarification,
            unresolved_pronouns=list(result.unresolved_pronouns),
            clarification_options=[
                str(x)
                for x in list(state.get("referenced_entities", []) or [])[:3]
                if isinstance(x, str) and x.strip()
            ],
        )


_context_resolver: ContextResolver | None = None


def get_context_resolver() -> ContextResolver:
    global _context_resolver
    if _context_resolver is None:
        _context_resolver = ContextResolver()
    return _context_resolver
