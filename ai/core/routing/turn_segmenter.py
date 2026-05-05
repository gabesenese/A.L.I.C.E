from __future__ import annotations

import re


class TurnSegmenter:
    PROJECT_MODE_PATTERN = re.compile(
        r"\b(?:ready to work on alice|let[' ]?s work on alice|let[' ]?s focus on my ai project|improve alice|make alice more agentic|let[' ]?s focus on making alice more agentic|make alice an operator|turn alice into an ai companion|keep moving alice toward agentic behavior|keep moving her toward operator behavior|compare alice to an agentic companion|fix alice[' ]?s routing|improve alice[' ]?s memory)\b",
        re.IGNORECASE,
    )

    @classmethod
    def looks_like_project_mode(cls, text: str) -> bool:
        return bool(cls.PROJECT_MODE_PATTERN.search(str(text or "")))
