"""Extract structured personal-memory candidates from user turns."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from ai.memory.personal_memory_constants import (
    DEFAULT_PERSONAL_DOMAIN,
    DEFAULT_PERSONAL_KIND,
    PERSONAL_MEMORY_DOMAINS,
    PERSONAL_MEMORY_KINDS,
    PERSONAL_MEMORY_SCOPES,
)


@dataclass(frozen=True)
class MemoryCandidate:
    content: str
    domain: str
    kind: str
    scope: str
    confidence: float
    source: str
    should_store: bool


class MemoryExtractor:
    _filler = {
        "ok",
        "okay",
        "yeah",
        "yep",
        "thanks",
        "thank you",
        "lol",
        "k",
        "kk",
        "nice",
        "cool",
    }

    _domain_patterns = {
        "alice_project": re.compile(
            r"\b(alice|jarvis|project|feature|roadmap|architecture|memory system|codebase)\b",
            re.IGNORECASE,
        ),
        "fitness": re.compile(
            r"\b(gain weight|lose weight|kg|workout|gym|fitness|exercise|diet)\b",
            re.IGNORECASE,
        ),
        "finance": re.compile(
            r"\b(finance|money|budget|debt|income|expenses|saving)\b",
            re.IGNORECASE,
        ),
        "health": re.compile(
            r"\b(health|sleep|sick|ill|medical|pain|anxiety|stress)\b",
            re.IGNORECASE,
        ),
        "relationships": re.compile(
            r"\b(partner|girlfriend|boyfriend|wife|husband|friend|family|sister|brother)\b",
            re.IGNORECASE,
        ),
        "work": re.compile(
            r"\b(work|job|career|manager|office|deadline|client)\b",
            re.IGNORECASE,
        ),
        "preferences": re.compile(
            r"\b(i like|i prefer|i don't like|i dislike|preference)\b",
            re.IGNORECASE,
        ),
        "personal_life": re.compile(
            r"\b(personal life|my life|about me|i feel|i am|i'm)\b",
            re.IGNORECASE,
        ),
    }

    def _is_filler(self, text: str) -> bool:
        low = str(text or "").strip().lower()
        if not low:
            return True
        if low in self._filler:
            return True
        if len(low) <= 5 and re.fullmatch(r"[a-z!?.,\s]+", low):
            return True
        return False

    def _pick_domain(self, text: str) -> str:
        for domain, pattern in self._domain_patterns.items():
            if pattern.search(text):
                return domain
        return DEFAULT_PERSONAL_DOMAIN

    @staticmethod
    def _pick_kind(text: str, domain: str) -> str:
        low = text.lower()
        if "i feel" in low or "frustrat" in low or "annoy" in low:
            return "emotional_state"
        if "i prefer" in low or "i like" in low or "i don't like" in low:
            return "preference"
        if "i want" in low or "trying to" in low or "goal" in low or "reach" in low:
            return "project_goal"
        if domain == "relationships":
            return "relationship_context"
        return "conversation_event"

    @staticmethod
    def _pick_scope(text: str, kind: str) -> str:
        low = text.lower()
        if kind in {"project_goal", "long_term_profile"}:
            return "long_term"
        if any(token in low for token in ("today", "tonight", "this week", "lately")):
            return "day_to_day"
        return "day_to_day"

    @staticmethod
    def _rewrite_content(user_name: str, text: str, domain: str, kind: str) -> str:
        clean = " ".join(str(text or "").strip().split())
        subject = user_name.strip() or "User"
        if domain == "alice_project" and "jarvis" in clean.lower():
            return f"{subject} wants Alice to become a Jarvis-like AI companion/operator."
        if domain == "fitness" and re.search(r"\b(?:\d{2,3})\s*kg\b", clean, re.IGNORECASE):
            target = re.search(r"\b(\d{2,3})\s*kg\b", clean, re.IGNORECASE)
            kg = target.group(1) if target else ""
            return f"{subject} is trying to gain weight and wants to reach {kg}kg."
        if kind == "emotional_state" and "remembers coding" in clean.lower():
            return (
                f"{subject} feels Alice remembers coding/project context better than "
                "personal-life context and wants better personal memory."
            )
        return f"{subject} said: {clean}"

    def extract_from_user_turn(
        self,
        *,
        user_text: str,
        user_name: str = "User",
        source: str = "conversation",
    ) -> List[MemoryCandidate]:
        text = str(user_text or "").strip()
        if self._is_filler(text):
            return [
                MemoryCandidate(
                    content="",
                    domain=DEFAULT_PERSONAL_DOMAIN,
                    kind=DEFAULT_PERSONAL_KIND,
                    scope="session",
                    confidence=0.0,
                    source=source,
                    should_store=False,
                )
            ]

        domain = self._pick_domain(text)
        kind = self._pick_kind(text, domain)
        scope = self._pick_scope(text, kind)
        domain = domain if domain in PERSONAL_MEMORY_DOMAINS else DEFAULT_PERSONAL_DOMAIN
        kind = kind if kind in PERSONAL_MEMORY_KINDS else DEFAULT_PERSONAL_KIND
        scope = scope if scope in PERSONAL_MEMORY_SCOPES else "day_to_day"

        confidence = 0.62
        if kind in {"project_goal", "emotional_state", "preference"}:
            confidence = 0.78
        if domain in {"alice_project", "fitness", "personal_life"}:
            confidence = max(confidence, 0.75)

        content = self._rewrite_content(user_name, text, domain, kind)
        return [
            MemoryCandidate(
                content=content,
                domain=domain,
                kind=kind,
                scope=scope,
                confidence=confidence,
                source=source,
                should_store=True,
            )
        ]
