from __future__ import annotations

import re


class TurnSegmenter:
    PROJECT_MODE_PATTERN = re.compile(
        r"\b(?:ready to work on alice|let[' ]?s work on alice|let[' ]?s focus on my ai project|improve alice|make alice more agentic|let[' ]?s focus on making alice more agentic)\b",
        re.IGNORECASE,
    )

    @classmethod
    def looks_like_project_mode(cls, text: str) -> bool:
        return bool(cls.PROJECT_MODE_PATTERN.search(str(text or "")))
