from __future__ import annotations

import re


_WEATHER_QUERY_PATTERNS = (
    r"\bwhat(?:'s| is) the weather\b",
    r"\bcheck (?:the )?weather\b",
    r"\bis it raining\b",
    r"\bdo i need (?:a )?(?:jacket|coat|umbrella)\b",
    r"\btemperature (?:today|outside)?\b",
    r"\bforecast\b",
)

_PROJECT_OBJECTIVE_PATTERNS = (
    r"\bready to work on alice\b",
    r"\bwork on alice\b",
    r"\bimprove alice\b",
    r"\bmake alice more agentic\b",
    r"\balice codebase\b",
    r"\bcontinue alice\b",
)

_EDUCATIONAL_PATTERNS = (
    r"\bbeginner\b",
    r"\bsimple\b",
    r"\bexplain how it works\b",
    r"\blearn something simple\b",
    r"\bresearch how\b",
    r"\bjust give me something\b",
    r"\bdoesn['o]?t matter what\b",
    r"\blearn more about\b",
    r"\blearn the basics\b",
    r"\blearn basics\b",
    r"\bteach me\b",
    r"\bgive me information\b",
    r"\bgive me some information\b",
    r"\btell me about\b",
    r"\bwhat is\b",
    r"\bwhat are\b",
    r"\bhow does .+ work\b",
    r"\bbasics of\b",
    r"\bintroduction to\b",
    r"\bintro to\b",
    r"\boverview of\b",
    r"\bindustry basics\b",
    r"\bagentic ai industry\b",
)

_IMPLEMENTATION_PATTERNS = (
    r"\blet[' ]?s implement\b",
    r"\bwork on alice\b",
    r"\binspect the codebase\b",
    r"\bfix alice\b",
    r"\badd this to alice\b",
    r"\bgive me a codex input\b",
)

_CODEBASE_ANALYZE_PATTERNS = (
    r"\banaly[sz]e .*code ?base\b",
    r"\banaly[sz]e .*codebase\b",
    r"\bable to analy[sz]e\b",
    r"\bcheck .*code ?base\b",
    r"\bgive me a codex input\b",
)


def is_explicit_weather_request(text: str) -> bool:
    low = str(text or "").lower()
    return any(re.search(p, low) for p in _WEATHER_QUERY_PATTERNS)


def resolve_dominant_intent_hint(text: str) -> str:
    low = str(text or "").lower().strip()
    if not low:
        return ""
    if any(re.search(p, low) for p in _CODEBASE_ANALYZE_PATTERNS):
        return "code:request"
    educational = any(re.search(p, low) for p in _EDUCATIONAL_PATTERNS)
    implementation = any(re.search(p, low) for p in _IMPLEMENTATION_PATTERNS)
    if educational and not implementation:
        return "conversation:educational_explain"
    if any(re.search(p, low) for p in _PROJECT_OBJECTIVE_PATTERNS):
        return "conversation:goal_statement"
    if is_explicit_weather_request(low):
        return "weather:current"
    return ""
