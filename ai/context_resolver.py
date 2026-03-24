"""Pre-NLP context resolver stage.

Pipeline target:
INPUT -> context resolver -> NLP classify -> planning -> tools
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from ai.ambiguity_detector import AmbiguityDetector
from ai.reference_resolver import ReferenceResolver


@dataclass
class ContextResolution:
    original_input: str
    rewritten_input: str
    resolved_bindings: Dict[str, str] = field(default_factory=dict)
    needs_clarification: bool = False
    unresolved_pronouns: List[str] = field(default_factory=list)
    clarification_options: List[str] = field(default_factory=list)


class ContextResolver:
    """Resolves references before NLP classification."""

    def __init__(self) -> None:
        self.reference_resolver = ReferenceResolver()
        self.ambiguity_detector = AmbiguityDetector()

    def resolve(self, user_input: str, state: Dict[str, object]) -> ContextResolution:
        result = self.reference_resolver.resolve(user_input=user_input, state=state)
        token_count = len((user_input or "").split())
        needs_clarification = self.ambiguity_detector.should_clarify(
            unresolved_pronouns=result.unresolved_pronouns,
            token_count=token_count,
        )

        return ContextResolution(
            original_input=user_input,
            rewritten_input=result.rewritten_input or user_input,
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
