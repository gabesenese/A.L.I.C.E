"""Detects ambiguous references and proposes clarification prompts."""

from __future__ import annotations

from typing import Iterable


class AmbiguityDetector:
    def should_clarify(
        self, *, unresolved_pronouns: Iterable[str], token_count: int
    ) -> bool:
        # Short pronoun-heavy follow-ups are the highest-risk misrouting shape.
        return bool(list(unresolved_pronouns)) and int(token_count or 0) <= 8
