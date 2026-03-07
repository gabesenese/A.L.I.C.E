"""
Response Quality Checker — automatic self-assessment after every interaction.

Catches the classes of error that previously required manual discovery:

1. DIRECTNESS — "should I wear a coat?" answered with "Bundle up" (no Yes/No)
2. REPETITION — temperature/location restated in every follow-up turn
3. VOCAB GAP  — user said "scarf" in a weather context; word not in fast-path list
4. UNNECESSARY_PLUGIN — plugin called when stored data was already available

Every issue is logged to  data/realtime_learning/quality_issues.jsonl
and available for analysis via  LearningInsights.generate_report().
"""

import json
import re
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from threading import Lock
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# ── Issue types ──────────────────────────────────────────────────────────────

ISSUE_DIRECTNESS         = "question_no_direct_answer"
ISSUE_REPETITION         = "info_repeated_in_followup"
ISSUE_VOCAB_GAP          = "domain_vocab_gap"
ISSUE_UNNECESSARY_PLUGIN = "unnecessary_plugin_call"

# ── Domain keyword reference lists ───────────────────────────────────────────
# These mirror the fast-path keyword lists in main.py / nlp_processor.py.
# When a word appears in a domain-intent context but is NOT here, it is a gap.

DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "weather": [
        "weather", "forecast", "temperature", "rain", "snow", "storm",
        "sunny", "cloud", "cold", "warm", "hot", "freeze",
        # clothing fast-path words
        "umbrella", "jacket", "coat", "layer", "wear", "bring",
        "outside", "go out", "scarf", "hat", "gloves", "boots",
        "sweater", "hoodie",
        # time cues
        "tomorrow", "tonight", "today", "week", "weekend",
        "monday", "tuesday", "wednesday", "thursday",
        "friday", "saturday", "sunday",
    ],
    "notes": [
        "note", "notes", "memo", "list", "lists", "todo", "task",
        "tasks", "write", "create", "add", "append", "read", "show",
        "delete", "remove", "find", "search", "open",
        # domain context words flagged by insights (added after first report)
        "created", "work", "tagged", "tag", "tags", "title",
        "coding", "project", "meeting", "idea", "ideas",
        "first", "last", "all", "my", "the",
    ],
    "email": [
        "email", "mail", "inbox", "message", "send", "compose",
        "draft", "reply", "read", "delete", "from", "subject",
    ],
    "reminder": [
        "remind", "reminder", "reminders", "alert", "notify",
        "schedule", "set", "cancel", "delete", "list", "upcoming",
        "pending",
    ],
    "music": [
        "music", "song", "songs", "play", "pause", "skip", "next",
        "playlist", "album", "artist",
    ],
    "calendar": [
        "calendar", "event", "events", "meeting", "meetings",
        "schedule", "appointment", "appointments", "agenda",
        "create", "set", "add", "tomorrow", "today", "next", "week",
    ],
}

# Stopwords to ignore when looking for vocab gaps
_STOPWORDS = {
    "i", "me", "my", "we", "our", "you", "your", "it", "its",
    "the", "a", "an", "and", "or", "but", "in", "on", "at",
    "to", "for", "of", "with", "from", "is", "are", "was",
    "be", "have", "has", "do", "does", "did", "will", "would",
    "can", "could", "should", "what", "when", "where", "who",
    "why", "how", "if", "that", "this", "there", "then", "about",
    "any", "all", "not", "no", "need", "want", "like", "just",
    "get", "go", "out", "up", "into", "over", "so", "also",
}

# Patterns that indicate a yes/no question
_YES_NO_QUESTION_PATTERNS = re.compile(
    r"\b(should i|do i need|do i have to|is it|will it|would i|"
    r"am i going to|need a|need to bring|need to wear)\b",
    re.IGNORECASE,
)

# Direct answer openers (yes/no/maybe)
_DIRECT_ANSWER_PATTERN = re.compile(
    r"^(yes|no|yeah|nope|yep|absolutely|definitely|probably|maybe|"
    r"not really|i'?d say|i would|you should|you don'?t)",
    re.IGNORECASE,
)


@dataclass
class QualityIssue:
    timestamp: str
    issue_type: str           # one of the ISSUE_* constants
    domain: str               # e.g. "weather"
    user_input: str
    alice_response: str
    intent: str
    detail: Dict[str, Any] = field(default_factory=dict)
    severity: str = "medium"  # low | medium | high

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ResponseQualityChecker:
    """
    Runs after every response and logs quality issues automatically.

    Usage (called from ALICE._store_interaction):
        checker.analyze(
            user_input=...,
            response=...,
            intent=...,
            plugin_called=...,         # name of plugin, or None
            had_stored_data=...,       # True if fast-path data was available
            previous_turn=...,         # dict from conversation_summary[-1], or None
        )
    """

    QUALITY_LOG = Path("data/realtime_learning/quality_issues.jsonl")

    def __init__(self):
        self.QUALITY_LOG.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        logger.debug("[QualityChecker] Initialized")

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(
        self,
        user_input: str,
        response: str,
        intent: str,
        plugin_called: Optional[str] = None,
        had_stored_data: bool = False,
        previous_turn: Optional[Dict[str, Any]] = None,
    ) -> List[QualityIssue]:
        """Run all checks and log any issues found. Returns list of issues."""
        issues: List[QualityIssue] = []
        domain = intent.split(":", 1)[0] if ":" in intent else intent

        issues += self._check_directness(user_input, response, intent, domain)
        issues += self._check_repetition(user_input, response, intent, domain, previous_turn)
        issues += self._check_vocab_gap(user_input, intent, domain)
        if plugin_called and had_stored_data:
            issues += self._check_unnecessary_plugin(user_input, intent, domain, plugin_called)

        if issues:
            self._log_issues(issues)
            for issue in issues:
                logger.debug(
                    "[QualityChecker] %s — %s (input: %s)",
                    issue.issue_type, issue.detail.get("reason", ""), user_input[:60],
                )

        return issues

    # ── Individual checks ─────────────────────────────────────────────────────

    def _check_directness(
        self, user_input: str, response: str, intent: str, domain: str
    ) -> List[QualityIssue]:
        """Flag when a yes/no question didn't get a yes/no answer."""
        if not _YES_NO_QUESTION_PATTERNS.search(user_input):
            return []
        if _DIRECT_ANSWER_PATTERN.match(response.strip("*_ ")):
            return []  # Good — starts with yes/no/maybe
        return [
            QualityIssue(
                timestamp=datetime.now().isoformat(),
                issue_type=ISSUE_DIRECTNESS,
                domain=domain,
                user_input=user_input,
                alice_response=response[:200],
                intent=intent,
                detail={
                    "reason": "yes/no question but response lacks direct Yes/No opener",
                    "response_start": response[:80],
                },
                severity="high",
            )
        ]

    def _check_repetition(
        self,
        user_input: str,
        response: str,
        intent: str,
        domain: str,
        previous_turn: Optional[Dict[str, Any]],
    ) -> List[QualityIssue]:
        """Flag when the response repeats the same key facts as the previous turn."""
        if not previous_turn:
            return []
        if not previous_turn.get("intent", "").startswith(domain):
            return []  # Different domain — not a follow-up

        prev_response = previous_turn.get("assistant", "")
        if not prev_response:
            return []

        # Find phrases of 3+ words that appear in both
        repeated = _shared_significant_phrases(prev_response, response)
        if not repeated:
            return []

        return [
            QualityIssue(
                timestamp=datetime.now().isoformat(),
                issue_type=ISSUE_REPETITION,
                domain=domain,
                user_input=user_input,
                alice_response=response[:200],
                intent=intent,
                detail={
                    "reason": "response repeats phrases from previous turn",
                    "repeated_phrases": repeated[:3],
                    "previous_response": prev_response[:120],
                },
                severity="medium",
            )
        ]

    def _check_vocab_gap(
        self, user_input: str, intent: str, domain: str
    ) -> List[QualityIssue]:
        """Flag domain-relevant words in user input that aren't in the keyword list."""
        known = set(DOMAIN_KEYWORDS.get(domain, []))
        if not known:
            return []

        input_words = set(re.findall(r"[a-z']+", user_input.lower())) - _STOPWORDS
        gap_words = input_words - known

        # Only flag words that appear near domain trigger words (contextual relevance)
        domain_triggers = {w for w in input_words if w in known}
        if not domain_triggers and not intent.startswith(domain):
            return []  # Unrelated query — skip

        if not gap_words:
            return []

        return [
            QualityIssue(
                timestamp=datetime.now().isoformat(),
                issue_type=ISSUE_VOCAB_GAP,
                domain=domain,
                user_input=user_input,
                alice_response="",
                intent=intent,
                detail={
                    "reason": f"words in {domain} context not in keyword list",
                    "gap_words": sorted(gap_words),
                    "known_triggers_present": sorted(domain_triggers),
                },
                severity="low",
            )
        ]

    def _check_unnecessary_plugin(
        self, user_input: str, intent: str, domain: str, plugin_called: str
    ) -> List[QualityIssue]:
        """Flag when a plugin was called despite stored data being available."""
        return [
            QualityIssue(
                timestamp=datetime.now().isoformat(),
                issue_type=ISSUE_UNNECESSARY_PLUGIN,
                domain=domain,
                user_input=user_input,
                alice_response="",
                intent=intent,
                detail={
                    "reason": "plugin called when stored data was available",
                    "plugin": plugin_called,
                    "user_words": list(
                        set(re.findall(r"[a-z']+", user_input.lower())) - _STOPWORDS
                    ),
                },
                severity="medium",
            )
        ]

    # ── Persistence ───────────────────────────────────────────────────────────

    def _log_issues(self, issues: List[QualityIssue]) -> None:
        try:
            with self._lock:
                with open(self.QUALITY_LOG, "a", encoding="utf-8") as f:
                    for issue in issues:
                        f.write(json.dumps(issue.to_dict()) + "\n")
        except Exception as e:
            logger.warning("[QualityChecker] Failed to log issues: %s", e)

    def load_recent(self, max_entries: int = 500) -> List[Dict[str, Any]]:
        """Load recent quality issues for analysis."""
        if not self.QUALITY_LOG.exists():
            return []
        issues = []
        try:
            with open(self.QUALITY_LOG, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        issues.append(json.loads(line))
        except Exception as e:
            logger.warning("[QualityChecker] Failed to load issues: %s", e)
        return issues[-max_entries:]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _shared_significant_phrases(a: str, b: str, min_words: int = 3) -> List[str]:
    """Return phrases of >= min_words tokens shared between two strings."""
    a_words = re.findall(r"[a-z'°]+", a.lower())
    b_text = b.lower()
    shared = []
    for i in range(len(a_words) - min_words + 1):
        phrase = " ".join(a_words[i : i + min_words])
        if phrase in b_text and not all(w in _STOPWORDS for w in phrase.split()):
            shared.append(phrase)
    # Deduplicate overlapping matches
    result = []
    for p in shared:
        if not any(p in existing for existing in result):
            result.append(p)
    return result


# ── Singleton ─────────────────────────────────────────────────────────────────

_checker: Optional[ResponseQualityChecker] = None


def get_quality_checker() -> ResponseQualityChecker:
    global _checker
    if _checker is None:
        _checker = ResponseQualityChecker()
    return _checker
