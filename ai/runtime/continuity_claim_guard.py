from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re
from typing import Any, Dict, List


# Alice has no camera and no microphone trained on the room. Anything about how the
# user looks, what they are wearing, or what is around them cannot be grounded by
# any amount of evidence, so it is removed on sight rather than checked against
# memory. Prompted for weather, she told a user he was "not exactly dressed for
# overcast skies".
_SENSORY_CLAIM_PATTERNS = (
    r"\byou'?re (?:not )?(?:exactly )?dressed\b",
    r"\byou are (?:not )?(?:exactly )?dressed\b",
    r"\byou'?re wearing\b",
    r"\bwhat you'?re wearing\b",
    r"\byou look (?:tired|great|good|rough|well|happy|sad|stressed)\b",
    r"\byou seem (?:tired|stressed|upset|happy|nervous)\b",
    r"\byour (?:face|outfit|posture|expression|desk|room|screen)\b",
    r"\bbehind you\b",
    r"\bi can see\b",
    r"\bfrom the looks of (?:it|you)\b",
)

_SENSORY_CLAIM_RE = re.compile("|".join(_SENSORY_CLAIM_PATTERNS), re.IGNORECASE)

# Words that start a sentence or are otherwise capitalised without naming anything.
_PROPER_NOUN_STOPWORDS = {
    "i",
    "i'm",
    "i've",
    "i'd",
    "i'll",
    "alice",
    "gabriel",
    "ok",
    "okay",
    "yes",
    "no",
    "not",
    # Pronouns and determiners, which are capitalised whenever they follow a colon
    # or a dash. Treating "You" as a name flagged every grounded memory recall.
    "you",
    "your",
    "yours",
    "we",
    "our",
    "ours",
    "they",
    "their",
    "them",
    "he",
    "him",
    "his",
    "she",
    "her",
    "hers",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "there",
    "here",
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
    "the",
    "and",
    "but",
    "still",
    "also",
    "just",
    "then",
    "than",
    "some",
    "any",
    "all",
    "both",
    "each",
    "every",
    "more",
    "most",
    "much",
    "let",
    "let's",
    "lets",
    "sure",
    "well",
    "maybe",
    "nothing",
    "something",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
}

_PROPER_NOUN_RE = re.compile(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{2,})\b")
_SECOND_PERSON_RE = re.compile(r"\byou\b|\byour\b|\byou'?re\b", re.IGNORECASE)
# Advice and hypotheticals put a name in front of the user without claiming
# anything about them. "You could use Postgres here" suggests; "are you heading
# to Oakville?" presumes. Only the second is an invented memory.
_ADVICE_RE = re.compile(
    r"\bif you\b|\byou(?:'d| could| should| can| might| may| would| will| need to| want to| could try)\b",
    re.IGNORECASE,
)


def _proper_nouns(sentence: str) -> List[str]:
    """Names mentioned inside a sentence, ignoring the capitalised first word."""
    body = str(sentence or "").strip()
    if not body:
        return []
    words = body.split()
    found: List[str] = []
    for word in words[1:]:
        token = word.strip(".,!?;:'\"()")
        if not token or not token[0].isupper() or len(token) < 3:
            continue
        if token.lower() in _PROPER_NOUN_STOPWORDS:
            continue
        if token.isupper():
            continue
        found.append(token.lower())
    return found


_CLAIM_PATTERNS = (
    r"\blast time we talked about\b",
    r"\bwe were discussing\b",
    r"\byou mentioned\b",
    r"\bwe left off on\b",
    r"\bi remember you were\b",
    r"\bour previous conversation was about\b",
    r"\bconversation history suggests\b",
    r"\bwhen we last spoke\b",
    r"\bas usual\b",
    r"\bstill on your mind\b",
    # Assertions about what the user is currently doing or feeling. These read as
    # recall and are invented just as easily: "you're still stuck on that routing
    # refactor" was produced for a user who had never mentioned a routing refactor.
    r"\byou'?re still\b",
    r"\byou'?ve been\b",
    r"\byou have been\b",
    r"\byou keep\b",
    r"\byou were (?:working|building|fixing|debugging|stuck|deep)\b",
    r"\bi can tell (?:you|that)\b",
    r"\bi know you'?(?:re|ve)\b",
    r"\bstill (?:stuck|grinding|buried|deep)\b",
    # Claims of having observed the user over time. Same fabrication, first person:
    # "I've noticed you're using SQLite again" about a user who never said so.
    r"\bi'?ve noticed\b",
    r"\bi notice\b",
    r"\bi'?ve seen you\b",
    r"\bi see you'?(?:re|ve)\b",
    r"\bevery time you\b",
)
_CLAIM_RE = re.compile("|".join(_CLAIM_PATTERNS), re.IGNORECASE)

# Words about habit and time claim knowledge of the user's routine only when the
# sentence is about the user. "You're up later than usual" asserts a routine nobody
# told her; "that build is taking longer than usual" is about the build, and deleting
# it cut the observation that the advice after it depended on. "As usual" stays a
# claim on its own: it asserts a shared routine whoever the subject is.
_HABIT_RE = re.compile(r"\bthan usual\b|\blately\b|\bthese days\b", re.IGNORECASE)

# Claims about an earlier occasion. Only stored memory can back these: something
# the user said a minute ago is no evidence of what was said last week.
_PRIOR_OCCASION_RE = re.compile(
    r"\blast time\b|\blast session\b|\bwhen we last spoke\b|\bwe left off\b|\bour previous conversation\b"
    r"|\bconversation history suggests\b|\byesterday\b|\blast week\b|\bthe other day\b",
    re.IGNORECASE,
)

# What she says when every sentence of a reply was an unsupported claim. The
# verifier gives the same line for the same failure (turn_orchestrator).
UNSUPPORTED_CLAIM_REPLY = "I don't have enough context to answer that confidently. Could you give me a bit more detail?"

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z0-9']+")
_STOPWORDS = {
    "the",
    "a",
    "an",
    "we",
    "you",
    "i",
    "me",
    "my",
    "our",
    "your",
    "was",
    "were",
    "are",
    "is",
    "to",
    "of",
    "about",
    "last",
    "time",
    "talked",
    "discussing",
    "mentioned",
    "remember",
    "previous",
    "conversation",
    "history",
    "suggests",
    "left",
    "off",
    "on",
    "and",
    "or",
    "in",
    "with",
    "for",
    "today",
    "day",
}

# Words too common to show that a claim is about something the user said. A
# transcript is full of them, so sharing one with a claim proves nothing.
_COMMON_WORDS = frozenset(
    (
        "about after again also been before being could does doing done even every first from going good "
        "have help here into just know like look make many might more most much need never next only other "
        "over really right same should since some something start still such sure take than that them then "
        "there these they thing things think this those time today very want well were what when where "
        "which while will with work would yeah your"
    ).split()
)


def _content_tokens(tokens: set[str]) -> set[str]:
    """The words in a token set that could name a topic."""
    content = set()
    for token in tokens:
        token = token[:-2] if token.endswith("'s") else token
        if len(token) >= 4 and "'" not in token and token not in _COMMON_WORDS:
            content.add(token)
    return content


@dataclass(frozen=True)
class ContinuityGuardResult:
    text: str
    detected_claims: List[str]
    supported_claims: List[str]
    unsupported_claims: List[str]
    evidence_sources: List[str]
    claim_topic_tokens: Dict[str, List[str]]
    evidence_topic_tokens: List[str]
    overlap_passed_by_claim: Dict[str, bool]
    support_reasons: Dict[str, List[str]]
    rejection_reasons: Dict[str, List[str]]
    recovery_applied: bool
    unsupported_continuity_claim: bool

    def metadata(self) -> Dict[str, Any]:
        return {
            "detected_claims": list(self.detected_claims),
            "supported_claims": list(self.supported_claims),
            "unsupported_claims": list(self.unsupported_claims),
            "evidence_sources": list(self.evidence_sources),
            "claim_topic_tokens": dict(self.claim_topic_tokens),
            "evidence_topic_tokens": list(self.evidence_topic_tokens),
            "overlap_required": True,
            "overlap_passed_by_claim": dict(self.overlap_passed_by_claim),
            "support_reasons": dict(self.support_reasons),
            "rejection_reasons": dict(self.rejection_reasons),
            "recovery_applied": bool(self.recovery_applied),
            "unsupported_continuity_claim": bool(self.unsupported_continuity_claim),
        }


def _parse_iso(value: str) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed
    except Exception:
        return None


def _source_name(item: Dict[str, Any]) -> str:
    ctx = dict(item.get("context") or {})
    return str(ctx.get("source") or item.get("source") or "").strip().lower()


def _tokens(text: str) -> set[str]:
    words = _WORD_RE.findall(str(text or "").lower())
    return {w for w in words if w and w not in _STOPWORDS}


def _claim_topic_tokens(claim: str) -> set[str]:
    cleaned = _HABIT_RE.sub(" ", _CLAIM_RE.sub(" ", str(claim or "").lower()))
    return _tokens(cleaned)


def _operator_state_topic_tokens(operator_state: Dict[str, Any]) -> set[str]:
    state = dict(operator_state or {})
    parts: List[str] = [
        str(state.get("active_objective") or ""),
        str(state.get("current_focus") or ""),
        str(state.get("current_step") or ""),
        str(state.get("next_recommended_action") or ""),
        str(state.get("last_inspected_file") or ""),
        str(state.get("last_success") or ""),
        str(state.get("last_failure") or ""),
    ]
    for key in ("current_plan", "files_inspected", "known_blockers"):
        value = state.get(key)
        if isinstance(value, list):
            parts.extend(str(v or "") for v in value)
    return _tokens(" ".join(parts))


def _memory_item_topic_text(item: Dict[str, Any]) -> str:
    ctx = dict(item.get("context") or {})
    tags = item.get("tags")
    tag_text = ""
    if isinstance(tags, list):
        tag_text = " ".join(str(t or "") for t in tags)
    return " ".join(
        [
            str(item.get("content") or ""),
            str(ctx.get("domain") or ""),
            str(ctx.get("kind") or ""),
            str(ctx.get("scope") or ""),
            tag_text,
        ]
    )


def _memory_topic_tokens(memory_items: List[Dict[str, Any]]) -> set[str]:
    corpus = " ".join(_memory_item_topic_text(i) for i in list(memory_items or []))
    return _tokens(corpus)


def _has_topic_overlap(claim_tokens: set[str], evidence_tokens: set[str]) -> bool:
    return bool(set(claim_tokens or set()).intersection(set(evidence_tokens or set())))


def _is_structured_memory(item: Dict[str, Any], *, min_confidence: float) -> bool:
    ctx = dict(item.get("context") or {})
    has_schema = all(str(ctx.get(k) or "").strip() for k in ("domain", "kind", "scope"))
    if not has_schema:
        return False
    confidence = float(ctx.get("confidence", item.get("importance", 0.0)) or 0.0)
    timestamp = str(ctx.get("timestamp") or item.get("timestamp") or "").strip()
    source = _source_name(item)
    return bool(
        confidence >= float(min_confidence)
        and timestamp
        and source
        and "vector" not in source
        and "embedding" not in source
    )


def _has_recent_session_evidence(items: List[Dict[str, Any]], *, now: datetime) -> bool:
    cutoff = now - timedelta(hours=6)
    for item in items:
        ctx = dict(item.get("context") or {})
        source = _source_name(item)
        turn_index = int(ctx.get("turn_index", -1) or -1)
        ts = _parse_iso(str(ctx.get("timestamp") or item.get("timestamp") or ""))
        if source in {"session", "session_recent", "conversation", "turn_memory"} and (
            turn_index >= 0 or (ts is not None and ts >= cutoff)
        ):
            return True
    return False


def _recent_session_items(items: List[Dict[str, Any]], *, now: datetime) -> List[Dict[str, Any]]:
    matched: List[Dict[str, Any]] = []
    cutoff = now - timedelta(hours=6)
    for item in list(items or []):
        ctx = dict(item.get("context") or {})
        source = _source_name(item)
        turn_index = int(ctx.get("turn_index", -1) or -1)
        ts = _parse_iso(str(ctx.get("timestamp") or item.get("timestamp") or ""))
        if source in {"session", "session_recent", "conversation", "turn_memory"} and (
            turn_index >= 0 or (ts is not None and ts >= cutoff)
        ):
            matched.append(item)
    return matched


def _has_active_objective(operator_state: Dict[str, Any]) -> bool:
    state = dict(operator_state or {})
    return bool(str(state.get("active_objective") or "").strip() and str(state.get("current_focus") or "").strip())


def assess_continuity_claims(
    *,
    text: str,
    memory_items: List[Dict[str, Any]],
    operator_state: Dict[str, Any] | None,
    min_structured_confidence: float = 0.7,
    evidence_text: str = "",
) -> ContinuityGuardResult:
    content = str(text or "").strip()
    if not content:
        return ContinuityGuardResult("", [], [], [], [], {}, [], {}, {}, {}, False, False)

    sentences = _SENTENCE_SPLIT.split(content)
    detected: List[str] = []
    supported: List[str] = []
    unsupported: List[str] = []
    evidence_sources: List[str] = []
    claim_topic_tokens_map: Dict[str, List[str]] = {}
    overlap_passed_by_claim: Dict[str, bool] = {}
    support_reasons: Dict[str, List[str]] = {}
    rejection_reasons: Dict[str, List[str]] = {}
    now = datetime.now(timezone.utc)
    all_items = list(memory_items or [])
    recent_items = _recent_session_items(all_items, now=now)
    structured_items = [
        row for row in all_items if _is_structured_memory(row, min_confidence=min_structured_confidence)
    ]
    state_tokens = _operator_state_topic_tokens(dict(operator_state or {}))
    operator_active = _has_active_objective(dict(operator_state or {}))

    for row in all_items:
        source = _source_name(row)
        if source and source not in evidence_sources:
            evidence_sources.append(source)
    if operator_active:
        evidence_sources.append("operator_state")

    # Names Alice is allowed to use: anything the user said in this conversation,
    # anything a tool returned this turn, plus stored memory and operator state.
    said_tokens = _tokens(str(evidence_text or ""))
    grounded_tokens = set(said_tokens)
    grounded_tokens |= state_tokens
    for item in all_items:
        grounded_tokens |= _tokens(str(item.get("content") or ""))
        grounded_tokens |= _tokens(_memory_item_topic_text(item))

    kept: List[str] = []
    for sentence in sentences:
        if _SENSORY_CLAIM_RE.search(sentence):
            claim = sentence.strip()
            detected.append(claim)
            unsupported.append(claim)
            claim_topic_tokens_map[claim] = []
            overlap_passed_by_claim[claim] = False
            rejection_reasons[claim] = ["no_sensor_for_this_claim"]
            continue

        # A name the user never used, in a sentence addressed to them, is invented.
        # "are you heading out for that drive to Oakville?" reads as recall and was
        # produced for a user who had never mentioned Oakville or a drive.
        if _SECOND_PERSON_RE.search(sentence) and not _ADVICE_RE.search(sentence):
            invented = [name for name in _proper_nouns(sentence) if name not in grounded_tokens]
            if invented:
                claim = sentence.strip()
                detected.append(claim)
                unsupported.append(claim)
                claim_topic_tokens_map[claim] = sorted(invented)
                overlap_passed_by_claim[claim] = False
                rejection_reasons[claim] = ["ungrounded_proper_noun"]
                continue

        if _CLAIM_RE.search(sentence) or (_HABIT_RE.search(sentence) and _SECOND_PERSON_RE.search(sentence)):
            claim = sentence.strip()
            detected.append(claim)
            claim_tokens = _claim_topic_tokens(claim)
            claim_topic_tokens_map[claim] = sorted(claim_tokens)
            reasons: List[str] = []

            # "You mentioned the tokenizer" right after the user did is a callback,
            # not an invention. The conversation itself is evidence, and it was not
            # consulted: turn memories carry no session source, so every in-session
            # callback was deleted.
            if not _PRIOR_OCCASION_RE.search(sentence) and _has_topic_overlap(
                _content_tokens(claim_tokens), _content_tokens(said_tokens)
            ):
                reasons.append("said_in_this_conversation")

            if recent_items:
                for item in recent_items:
                    if _has_topic_overlap(claim_tokens, _tokens(str(item.get("content") or ""))):
                        reasons.append("recent_session_overlap")
                        break

            if operator_active and state_tokens and _has_topic_overlap(claim_tokens, state_tokens):
                reasons.append("active_objective_overlap")

            if structured_items:
                for item in structured_items:
                    if _has_topic_overlap(claim_tokens, _tokens(_memory_item_topic_text(item))):
                        reasons.append("structured_memory_overlap")
                        break

            if reasons:
                supported.append(claim)
                overlap_passed_by_claim[claim] = True
                support_reasons[claim] = reasons
                kept.append(sentence)
            else:
                unsupported.append(claim)
                overlap_passed_by_claim[claim] = False
                reasons = []
                if not claim_tokens:
                    reasons.append("no_claim_topic_tokens")
                if not recent_items:
                    reasons.append("no_recent_session_evidence")
                elif recent_items:
                    reasons.append("no_recent_session_topic_overlap")
                if not operator_active:
                    reasons.append("no_active_operator_objective")
                elif operator_active:
                    reasons.append("no_operator_state_topic_overlap")
                if not structured_items:
                    reasons.append("no_structured_memory_evidence")
                elif structured_items:
                    reasons.append("no_structured_memory_topic_overlap")
                rejection_reasons[claim] = reasons
        else:
            kept.append(sentence)

    cleaned = " ".join(part.strip() for part in kept if part.strip()).strip()
    recovery_applied = bool(unsupported)
    if recovery_applied and not cleaned:
        # This used to be "I am here. No active task is loaded yet...", a boot
        # banner in the middle of a conversation, in place of an answer.
        cleaned = UNSUPPORTED_CLAIM_REPLY

    return ContinuityGuardResult(
        text=cleaned,
        detected_claims=detected,
        supported_claims=supported,
        unsupported_claims=unsupported,
        evidence_sources=evidence_sources,
        claim_topic_tokens=claim_topic_tokens_map,
        evidence_topic_tokens=sorted(_memory_topic_tokens(recent_items + structured_items).union(state_tokens)),
        overlap_passed_by_claim=overlap_passed_by_claim,
        support_reasons=support_reasons,
        rejection_reasons=rejection_reasons,
        recovery_applied=recovery_applied,
        unsupported_continuity_claim=bool(unsupported),
    )
