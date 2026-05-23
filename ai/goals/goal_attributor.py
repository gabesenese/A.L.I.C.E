"""Per-turn goal attribution and completion detection — Foundation 3.

After every turn, attributes the conversation to the most relevant active
goal using token-overlap (Jaccard similarity). No LLM calls.

Also detects explicit completion signals in user input and auto-completes
the attributed goal.
"""

from __future__ import annotations

import re
from typing import Optional, Set

_STOPWORDS: Set[str] = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "as",
    "is",
    "was",
    "are",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "can",
    "this",
    "that",
    "it",
    "its",
    "i",
    "we",
    "you",
    "he",
    "she",
    "they",
    "my",
    "your",
    "our",
    "their",
    "what",
    "which",
    "who",
    "how",
    "when",
    "where",
    "why",
    "let",
    "just",
    "also",
    "now",
    "get",
    "got",
    "make",
    "use",
}

_CODE_INTENTS = {"code", "tool", "plugin", "file", "notes", "search"}

_COMPLETION_SIGNALS = re.compile(
    r"\b(got\s+it\s+working|that'?s\s+done|all\s+done|fixed|resolved|shipped|"
    r"closed|finished|works\s+now|working\s+now|done\s+with|completed|wrapped\s+up)\b",
    re.IGNORECASE,
)

_MIN_SIMILARITY = 0.15
_COMPLETION_SIMILARITY = 0.10  # lower threshold when completion signal present


def _tokenize(text: str) -> Set[str]:
    tokens = re.findall(r"\b[a-z][a-z0-9_]{2,}\b", text.lower())
    return {t for t in tokens if t not in _STOPWORDS}


def _similarity(turn: Set[str], goal: Set[str]) -> float:
    """Max of Jaccard and turn-recall (overlap / |turn|).

    Turn-recall handles long goal descriptions gracefully — a 9-token turn
    that matches 3 of a 40-token goal description scores 3/9=0.33 on recall
    vs. 3/46=0.065 on Jaccard. We take the max.
    """
    if not turn or not goal:
        return 0.0
    inter = len(turn & goal)
    if inter == 0:
        return 0.0
    jaccard = inter / len(turn | goal)
    recall = inter / len(turn)
    return max(jaccard, recall)


def _intent_boost(intent: str, goal_tokens: Set[str]) -> float:
    intent_root = str(intent or "").split(":")[0].lower()
    boost = 0.0
    if intent_root in _CODE_INTENTS:
        code_terms = {
            "code",
            "function",
            "class",
            "file",
            "bug",
            "error",
            "implement",
            "refactor",
            "test",
            "build",
            "fix",
            "route",
            "routing",
            "pipeline",
            "module",
            "system",
        }
        if goal_tokens & code_terms:
            boost += 0.05
    return boost


def attribute_turn_to_goal(
    user_input: str,
    alice_response: str,
    intent: str,
    session_id: str = "",
) -> Optional[str]:
    """Find the active goal most related to this turn. Mark it worked if found.

    Returns the matched goal_id, or None.
    """
    try:
        from ai.goals.goal_engine import get_goal_engine

        engine = get_goal_engine()
        active = engine.active()
        if not active:
            return None

        turn_text = f"{user_input} {alice_response}"
        turn_tokens = _tokenize(turn_text)
        if not turn_tokens:
            return None

        has_completion = bool(_COMPLETION_SIGNALS.search(user_input))
        threshold = _COMPLETION_SIMILARITY if has_completion else _MIN_SIMILARITY

        best_goal_id: Optional[str] = None
        best_score: float = 0.0

        for goal in active:
            goal_text = f"{goal.description} {goal.next_action or ''}"
            goal_tokens = _tokenize(goal_text)
            score = _similarity(turn_tokens, goal_tokens) + _intent_boost(
                intent, goal_tokens
            )
            if score > best_score:
                best_score = score
                best_goal_id = goal.goal_id

        if best_score < threshold or best_goal_id is None:
            return None

        if has_completion:
            engine.complete(
                best_goal_id,
                session_id=session_id,
                note="completion signal in user input",
            )
        else:
            engine.mark_worked(best_goal_id, session_id=session_id, intent=intent)

        return best_goal_id

    except Exception:
        return None


def get_attributed_goal_context(
    user_input: str,
    intent: str,
) -> Optional[str]:
    """Return a one-line "Working toward: ..." string for companion context injection.

    Uses a read-only scoring pass — does NOT mark goals worked (that's done
    post-turn by attribute_turn_to_goal).
    """
    try:
        from ai.goals.goal_engine import get_goal_engine

        engine = get_goal_engine()
        active = engine.active()
        if not active:
            return None

        turn_tokens = _tokenize(user_input)
        if not turn_tokens:
            return None

        best_goal = None
        best_score = 0.0

        for goal in active:
            goal_text = f"{goal.description} {goal.next_action or ''}"
            goal_tokens = _tokenize(goal_text)
            score = _similarity(turn_tokens, goal_tokens) + _intent_boost(
                intent, goal_tokens
            )
            if score > best_score:
                best_score = score
                best_goal = goal

        if best_score < 0.25 or best_goal is None:
            return None

        pct = best_goal.progress_pct()
        days = best_goal.days_since_worked()
        parts = [f"Working toward: {best_goal.description[:70]}"]
        if pct > 0:
            parts.append(f"{pct:.0%} complete")
        if days is not None and days >= 1:
            parts.append(f"last touched {int(days)}d ago")
        return " — ".join(parts) if len(parts) > 1 else parts[0]

    except Exception:
        return None
