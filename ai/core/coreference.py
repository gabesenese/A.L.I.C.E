"""
A.L.I.C.E. Advanced Coreference Resolution Engine
====================================================
Entity chain tracking across multiple dialogue turns with typed resolution.

Architecture
------------
EntityChain  — ordered list of EntityMention objects (sliding window = 12)
DialogueMemory — per-session store of past turns, used for resolution lookup
AdvancedCoreferenceResolver — main engine

Supported reference types
--------------------------
| Pattern                    | Resolution type  | Example                        |
|----------------------------|------------------|--------------------------------|
| it / that / this / the one | PRONOUN_GENERIC  | "open it" → last note           |
| the note / the file        | DOMAIN_PRONOUN   | "delete the note"              |
| the X one / the X note     | DESCRIPTIVE_REF  | "the work one"                 |
| the first/second/Nth one   | ORDINAL_REF      | "open the second one"          |
| the one tagged X / by X    | ATTRIBUTE_REF    | "the one tagged work"          |
| the last / the previous     | RECENCY_REF      | "edit the last one"            |

Usage
-----
>>> from ai.core.coreference import AdvancedCoreferenceResolver, DialogueMemory
>>> mem = DialogueMemory()
>>> resolver = AdvancedCoreferenceResolver(mem)
>>> # After a search returned ["meeting notes", "shopping list"]
>>> mem.record_result_set("RESULT_SET", ["meeting notes", "shopping list"])
>>> resolver.resolve("open the first one", {})
ResolvedText(text='open "meeting notes"', resolved=True, entity_type='ORDINAL_REF', ...)
"""

from __future__ import annotations

import re
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EntityMention:
    """A single entity observed in one dialogue turn."""

    entity_type: (
        str  # NOTE_REF | EMAIL_REF | EVENT_REF | RESULT_SET | SONG_REF | PERSON_REF
    )
    value: Any  # str for single entity, list for RESULT_SET
    turn_index: int  # which dialogue turn
    plugin: str = ""  # which plugin produced this entity
    tags: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResolvedText:
    """Output of the coreference resolver."""

    text: str  # resolved utterance
    resolved: bool  # was any substitution made?
    entity_type: Optional[str]  # type of the resolved entity
    resolved_value: Any  # the actual value substituted
    confidence: float  # 0.0–1.0
    substitution_map: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dialogue Memory
# ---------------------------------------------------------------------------


class DialogueMemory:
    """
    Sliding-window store of entity mentions across turns.
    Provides typed lookups used by the resolver.
    """

    WINDOW = 12  # keep last N turns' entities

    def __init__(self):
        self._chain: deque[EntityMention] = deque(maxlen=self.WINDOW * 4)
        self._turn: int = 0

    # ── writers ────────────────────────────────────────────────────────

    def new_turn(self) -> None:
        self._turn += 1

    def record(
        self,
        entity_type: str,
        value: Any,
        plugin: str = "",
        tags: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        mention = EntityMention(
            entity_type=entity_type,
            value=value,
            turn_index=self._turn,
            plugin=plugin,
            tags=tags or [],
            extra=extra or {},
        )
        self._chain.append(mention)
        logger.debug(
            "[COREF-MEM] recorded %s=%r turn=%d", entity_type, value, self._turn
        )

    def record_result_set(
        self, entity_type: str, values: List[str], plugin: str = "notes"
    ) -> None:
        """Convenience: record a list result (e.g. search results)."""
        self.record(
            entity_type="RESULT_SET",
            value=values,
            plugin=plugin,
            extra={"original_type": entity_type},
        )

    def update_from_nlp_result(
        self,
        intent: str,
        slots: Dict[str, Any],
        response_titles: Optional[List[str]] = None,
    ) -> None:
        """
        Called after each NLP+plugin cycle to update the entity chain from
        what was just processed.
        """
        self.new_turn()
        plugin = intent.split(":")[0] if ":" in intent else intent

        # Track result set from search responses
        if response_titles:
            self.record_result_set("RESULT_SET", response_titles, plugin=plugin)

        # Track individual note references
        title = slots.get("title") or slots.get("note_ref") or slots.get("query")
        if title and isinstance(title, str):
            self.record("NOTE_REF", title, plugin=plugin)

        # Track tags
        tags = slots.get("tags")
        if tags:
            tag_list = tags if isinstance(tags, list) else [tags]
            self.record("TAGS", tag_list, plugin=plugin)

    # ── readers ────────────────────────────────────────────────────────

    def last_of_type(self, *entity_types: str) -> Optional[EntityMention]:
        """Return the most recent mention of any of the given types."""
        for mention in reversed(self._chain):
            if mention.entity_type in entity_types:
                return mention
        return None

    def all_of_type(self, *entity_types: str) -> List[EntityMention]:
        return [m for m in self._chain if m.entity_type in entity_types]

    def get_result_set(self) -> List[str]:
        """Return the most recent RESULT_SET list, or []."""
        m = self.last_of_type("RESULT_SET")
        if m and isinstance(m.value, list):
            return m.value
        return []

    def entity_chain_dict(self) -> List[Dict[str, Any]]:
        """Serialize chain for use in context dict passed to slot filler."""
        result: List[Dict[str, Any]] = []
        for m in reversed(self._chain):
            result.append(
                {
                    "type": m.entity_type,
                    "value": m.value,
                    "plugin": m.plugin,
                    "turn": m.turn_index,
                    "tags": m.tags,
                }
            )
        return result

    def clear(self) -> None:
        self._chain.clear()
        self._turn = 0


# ---------------------------------------------------------------------------
# Ordinal words → indices
# ---------------------------------------------------------------------------

_ORDINAL_MAP: Dict[str, int] = {
    "first": 0,
    "1st": 0,
    "one": 0,
    "second": 1,
    "2nd": 1,
    "two": 1,
    "third": 2,
    "3rd": 2,
    "three": 2,
    "fourth": 3,
    "4th": 3,
    "four": 3,
    "fifth": 4,
    "5th": 4,
    "five": 4,
    "sixth": 5,
    "6th": 5,
    "six": 5,
    "seventh": 6,
    "7th": 6,
    "eighth": 7,
    "8th": 7,
    "last": -1,
    "previous": -1,
    "latest": -1,
}

# Pronoun groups
_GENERIC_PRONOUNS = frozenset(
    ["it", "that", "this", "the result", "this one", "that one"]
)
_DOMAIN_PHRASES = frozenset(
    [
        "the note",
        "the file",
        "the message",
        "the event",
        "the email",
        "the song",
        "the track",
        "the entry",
    ]
)


# ---------------------------------------------------------------------------
# Advanced Coreference Resolver
# ---------------------------------------------------------------------------


class AdvancedCoreferenceResolver:
    """
    Resolve pronoun/referential expressions in an utterance using
    DialogueMemory.

    Resolution order (highest priority first):
    1. ORDINAL_REF     — "the first one", "the second note"
    2. ATTRIBUTE_REF   — "the one tagged work", "the one from yesterday"
    3. RECENCY_REF     — "the last one", "the previous"
    4. DESCRIPTIVE_REF — "the work note", "the meeting one"
    5. DOMAIN_PRONOUN  — "the note", "the file"
    6. PRONOUN_GENERIC — "it", "that", "this"
    """

    # ── compiled patterns ──────────────────────────────────────────────

    # Ordinal: "the second one", "the 3rd note", "note number 2"
    _RE_ORDINAL = re.compile(
        r"\bthe\s+("
        + "|".join(re.escape(k) for k in _ORDINAL_MAP)
        + r")\s+(?:one|note|file|result)?\b"
        r"|(?:note|result)\s+(?:number\s+)?#?(\d+)\b",
        re.IGNORECASE,
    )

    # Attribute: "the one tagged X", "the one from yesterday"
    _RE_ATTRIBUTE_TAG = re.compile(r"\bthe\s+one\s+tagged\s+([\w,\s]+?)(?:\s|$)", re.I)
    _RE_ATTRIBUTE_WORD = re.compile(
        r"\bthe\s+one\s+(?:about|with|from|mentioning)\s+(.+?)(?:\s|$)", re.I
    )

    # Recency: "the last one", "the previous note"
    _RE_RECENCY = re.compile(
        r"\bthe\s+(?:last|latest|most\s+recent|previous)\s+(?:one|note|file|result)?\b",
        re.I,
    )

    # Descriptive: "the work one", "the meeting note"
    _RE_DESCRIPTIVE = re.compile(r"\bthe\s+(\w+)\s+(?:one|note|file|result)\b", re.I)

    # Domain pronouns
    _RE_DOMAIN = re.compile(
        r"\b(" + "|".join(re.escape(p) for p in _DOMAIN_PHRASES) + r")\b", re.I
    )

    # Generic pronouns
    _RE_PRONOUN = re.compile(
        r"\b(" + "|".join(re.escape(p) for p in _GENERIC_PRONOUNS) + r")\b", re.I
    )

    # Idiomatic vague phrases that must NOT trigger coref resolution
    _RE_NO_RESOLVE = re.compile(
        r"\b(do\s+that\s+thing|that\s+thing|this\s+thing|what\s+about\s+that|"
        r"who\s+is\s+that|things?\s+like\s+that|something\s+like\s+that)\b",
        re.IGNORECASE,
    )

    def __init__(self, memory: Optional[DialogueMemory] = None):
        self.memory = memory or DialogueMemory()

    def resolve(self, text: str, context: Dict[str, Any]) -> ResolvedText:
        """
        Main entry point.  Returns ResolvedText — always contains final `text`
        (possibly identical to input if nothing was resolved).
        """
        # Skip resolution for idiomatic/vague phrases
        if self._RE_NO_RESOLVE.search(text):
            return ResolvedText(
                text=text,
                resolved=False,
                entity_type=None,
                resolved_value=None,
                confidence=0.0,
            )
        original = text
        text, sub_map, etype, evalue, conf = self._apply_resolutions(text, context)

        resolved = text != original
        return ResolvedText(
            text=text,
            resolved=resolved,
            entity_type=etype,
            resolved_value=evalue,
            confidence=conf,
            substitution_map=sub_map,
        )

    def resolve_text(self, text: str, context: Dict[str, Any]) -> str:
        """Convenience: return just the resolved text string."""
        return self.resolve(text, context).text

    # ------------------------------------------------------------------
    # Internal resolution pipeline
    # ------------------------------------------------------------------

    def _apply_resolutions(self, text: str, ctx: Dict[str, Any]):
        sub_map: Dict[str, str] = {}
        etype: Optional[str] = None
        evalue: Any = None
        conf: float = 0.0

        result_set = self.memory.get_result_set()

        # 1. ORDINAL_REF
        m = self._RE_ORDINAL.search(text)
        if m:
            ordinal_word = (m.group(1) or "").lower()
            raw_num = m.group(2)
            idx: Optional[int] = None
            if ordinal_word:
                idx = _ORDINAL_MAP.get(ordinal_word)
            elif raw_num:
                i = int(raw_num) - 1
                idx = max(0, i)

            if idx is not None and result_set:
                try:
                    if idx == -1:
                        target = result_set[-1]
                    else:
                        target = result_set[idx]
                    replacement = f'"{target}"'
                    old = m.group(0)
                    text = text[: m.start()] + replacement + text[m.end() :]
                    sub_map[old] = replacement
                    etype, evalue, conf = "ORDINAL_REF", target, 0.92
                    logger.info("[COREF] ORDINAL '%s' -> '%s'", old, replacement)
                except IndexError:
                    pass

        # 2. ATTRIBUTE_REF by tag
        m = self._RE_ATTRIBUTE_TAG.search(text)
        if m and not etype:
            tag_query = m.group(1).strip().lower()
            # Find the first note in results that has this tag (if we have tag info)
            candidate = self._find_by_attribute(tag_query, "tags", result_set)
            if candidate:
                old = m.group(0)
                replacement = f'"{candidate}"'
                text = text.replace(old, replacement, 1)
                sub_map[old] = replacement
                etype, evalue, conf = "ATTRIBUTE_REF", candidate, 0.85
                logger.info("[COREF] ATTRIBUTE-TAG '%s' -> '%s'", old, replacement)

        # 3. ATTRIBUTE_REF by keyword
        m = self._RE_ATTRIBUTE_WORD.search(text)
        if m and not etype:
            keyword = m.group(1).strip().lower()
            candidate = self._find_by_keyword(keyword, result_set)
            if candidate:
                old = m.group(0)
                replacement = f'"{candidate}"'
                text = text.replace(old, replacement, 1)
                sub_map[old] = replacement
                etype, evalue, conf = "ATTRIBUTE_REF", candidate, 0.82
                logger.info("[COREF] ATTRIBUTE-KW '%s' -> '%s'", old, replacement)

        # 4. RECENCY_REF
        m = self._RE_RECENCY.search(text)
        if m and not etype:
            # last in result_set, or last NOTE_REF in chain
            candidate = None
            if result_set:
                candidate = result_set[-1]
            else:
                mention = self.memory.last_of_type("NOTE_REF")
                if mention:
                    candidate = str(mention.value)
            if candidate:
                old = m.group(0)
                replacement = f'"{candidate}"'
                text = text.replace(old, replacement, 1)
                sub_map[old] = replacement
                etype, evalue, conf = "RECENCY_REF", candidate, 0.88
                logger.info("[COREF] RECENCY '%s' -> '%s'", old, replacement)

        # 5. DESCRIPTIVE_REF ("the work one")
        m = self._RE_DESCRIPTIVE.search(text)
        if m and not etype:
            descriptor = m.group(1).lower()
            candidate = self._find_by_keyword(descriptor, result_set)
            if candidate:
                old = m.group(0)
                replacement = f'"{candidate}"'
                text = text.replace(old, replacement, 1)
                sub_map[old] = replacement
                etype, evalue, conf = "DESCRIPTIVE_REF", candidate, 0.80
                logger.info("[COREF] DESCRIPTIVE '%s' -> '%s'", old, replacement)

        # 6. DOMAIN_PRONOUN ("the note", "the file")
        m = self._RE_DOMAIN.search(text)
        if m and not etype:
            candidate = self._last_note_ref(ctx)
            if candidate:
                old = m.group(0)
                replacement = f'"{candidate}"'
                text = text.replace(old, replacement, 1)
                sub_map[old] = replacement
                etype, evalue, conf = "DOMAIN_PRONOUN", candidate, 0.82
                logger.info("[COREF] DOMAIN '%s' -> '%s'", old, replacement)

        # 7. PRONOUN_GENERIC ("it", "that", "this")
        m = self._RE_PRONOUN.search(text)
        if m and not etype:
            candidate = self._last_note_ref(ctx)
            if candidate:
                old = m.group(0)
                replacement = f'"{candidate}"'
                text = text.replace(old, replacement, 1)
                sub_map[old] = replacement
                etype, evalue, conf = "PRONOUN_GENERIC", candidate, 0.75
                logger.info("[COREF] PRONOUN '%s' -> '%s'", old, replacement)

        return text, sub_map, etype, evalue, conf

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def _last_note_ref(self, ctx: Dict[str, Any]) -> Optional[str]:
        """Best guess at the most recently discussed note title."""
        # DialogueMemory has highest fidelity
        mention = self.memory.last_of_type("NOTE_REF")
        if mention:
            return str(mention.value)
        # Fall back to context dict
        title = ctx.get("last_note_title") or ctx.get("last_entities", {}).get("title")
        if title:
            return str(title)
        # Fall back to result set – most recent single note
        rs = self.memory.get_result_set()
        if len(rs) == 1:
            return rs[0]
        return None

    def _find_by_attribute(
        self, attribute: str, attribute_type: str, result_set: List[str]
    ) -> Optional[str]:
        """Find a note in the entity chain that matches an attribute."""
        for mention in reversed(self.memory.all_of_type("NOTE_REF")):
            if attribute_type == "tags":
                note_tags = [t.lower() for t in mention.tags]
                if attribute.lower() in note_tags:
                    return str(mention.value)
        # As fallback search result_set by substring
        for title in result_set:
            if attribute in title.lower():
                return title
        return None

    def _find_by_keyword(self, keyword: str, result_set: List[str]) -> Optional[str]:
        """Find a result_set entry whose title contains the keyword."""
        for title in result_set:
            if keyword in title.lower():
                return title
        # Also check entity chain
        for mention in reversed(self.memory.all_of_type("NOTE_REF")):
            if keyword in str(mention.value).lower():
                return str(mention.value)
        return None


# ---------------------------------------------------------------------------
# Compatibility shim: drop-in replacement for the old CoreferenceResolver
# ---------------------------------------------------------------------------


class LegacyCoreferenceResolverCompat:
    """
    Wraps AdvancedCoreferenceResolver with the old interface:
      resolve(text, context) -> str
    Used by nlp_processor.py until it's fully migrated.
    """

    def __init__(self, memory: Optional[DialogueMemory] = None):
        self._memory = memory or DialogueMemory()
        self._engine = AdvancedCoreferenceResolver(self._memory)

    @property
    def memory(self) -> DialogueMemory:
        return self._memory

    def resolve(self, text: str, context: Any) -> str:
        """Compatible with old CoreferenceResolver.resolve(text, ConversationContext)."""
        # Build a lightweight dict from ConversationContext if needed
        ctx: Dict[str, Any] = {}
        if hasattr(context, "last_entities"):
            ctx["last_entities"] = context.last_entities or {}
        if hasattr(context, "mentioned_notes"):
            notes = list(context.mentioned_notes)
            if notes:
                ctx["last_note_title"] = str(notes[-1])
                # Sync result set into memory if not already there
                if not self._memory.get_result_set() and notes:
                    self._memory.record_result_set(
                        "RESULT_SET", [str(n) for n in notes]
                    )
        return self._engine.resolve_text(text, ctx)
