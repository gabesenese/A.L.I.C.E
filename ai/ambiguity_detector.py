"""Detects ambiguous references and proposes clarification prompts."""

from __future__ import annotations

import re
from typing import Iterable


class AmbiguityDetector:
    _WEATHER_QUERY_RE = re.compile(
        r"\b("
        r"weather|forecast|rain|raining|snow|snowing|drizzle|shower|storm|thunder|"
        r"temperature|humid|humidity|wind|windy|sunny|cloudy|overcast|umbrella|outside"
        r")\b",
        re.IGNORECASE,
    )

    def _looks_like_weather_pronoun_query(
        self, *, user_input: str, unresolved_pronouns: Iterable[str]
    ) -> bool:
        low = str(user_input or "").lower()
        if not low:
            return False
        pronouns = {str(p).strip().lower() for p in unresolved_pronouns if str(p).strip()}
        # Only bypass when the ambiguity is likely a weather placeholder pronoun.
        if not pronouns or not pronouns.issubset({"it", "its", "it's", "that", "this"}):
            return False
        return bool(self._WEATHER_QUERY_RE.search(low))

    def should_clarify(
        self,
        *,
        unresolved_pronouns: Iterable[str],
        token_count: int,
        user_input: str = "",
    ) -> bool:
        unresolved = list(unresolved_pronouns)
        if not unresolved:
            return False
        if self._looks_like_weather_pronoun_query(
            user_input=user_input, unresolved_pronouns=unresolved
        ):
            return False
        # Short pronoun-heavy follow-ups are the highest-risk misrouting shape.
        return int(token_count or 0) <= 8
