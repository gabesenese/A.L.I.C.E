from __future__ import annotations

import re


class TurnSegmenter:
    PROJECT_MODE_PATTERN = re.compile(r"\b(?:ready to work on alice|let[' ]?s work on alice|let[' ]?s focus on my ai project|compare alice to jarvis|improve alice)\b", re.IGNORECASE)

    @classmethod
    def looks_like_project_mode(cls, text: str) -> bool:
        return bool(cls.PROJECT_MODE_PATTERN.search(str(text or "")))
