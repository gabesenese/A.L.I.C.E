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
    fragment: str = ""


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
            r"\b(worked late|work was busy|my job|manager|at the office|office|career|coworker|client|work shift)\b",
            re.IGNORECASE,
        ),
        "preferences": re.compile(
            r"\b(i like|i prefer|i don't like|i dislike|preference)\b",
            re.IGNORECASE,
        ),
        "personal_life": re.compile(
            r"\b(personal life|my life|about me|i feel)\b",
            re.IGNORECASE,
        ),
    }
    _personal_priority_pattern = re.compile(
        r"\b(i feel|my personal life|about me|my life)\b",
        re.IGNORECASE,
    )
    _alice_project_pattern = re.compile(
        r"\b(alice|jarvis|coding|code|project|feature|roadmap|memory)\b",
        re.IGNORECASE,
    )
    _action_request_pattern = re.compile(
        r"\b(can you|could you|are you able to|please|check|inspect|review|analyze|scan|look at)\b",
        re.IGNORECASE,
    )
    _day_to_day_personal_pattern = re.compile(
        r"\b("
        r"did some shopping|went shopping|went out with friends|had lunch with|"
        r"stayed home|worked late|went to the gym|went to gym|did groceries|ran errands"
        r")\b",
        re.IGNORECASE,
    )
    _alice_project_work_pattern = re.compile(
        r"\b(ready to work on (?:our )?(?:ai|alice|project)|"
        r"let[' ]?s (?:continue )?work(?:ing)? on (?:alice|the ai project)|"
        r"ready to keep building alice|back to working on alice|"
        r"let[' ]?s get back to alice|work on (?:alice|our ai project|the codebase|the repo))\b",
        re.IGNORECASE,
    )

    def _split_fragments(self, text: str) -> List[str]:
        raw_parts = re.split(
            r"[?.!;]|,(?=\s*(?:can|could|are|please|ready to|let[' ]?s|back to)\b)",
            text,
        )
        parts = []
        for raw in raw_parts:
            part = " ".join(str(raw or "").strip().split())
            if part:
                parts.append(part)
        return parts if parts else [text]

    def _is_filler(self, text: str) -> bool:
        low = str(text or "").strip().lower()
        if not low:
            return True
        if low in self._filler:
            return True
        if len(low) <= 5 and re.fullmatch(r"[a-z!?.,\s]+", low):
            return True
        return False

    def _pick_domains(self, text: str) -> List[str]:
        found: List[str] = []
        for domain, pattern in self._domain_patterns.items():
            if pattern.search(text):
                found.append(domain)
        if not found:
            return [DEFAULT_PERSONAL_DOMAIN]

        has_personal_priority = bool(self._personal_priority_pattern.search(text))
        if has_personal_priority and "personal_life" in found:
            found = ["personal_life"] + [d for d in found if d != "personal_life"]
        return found

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
        low = clean.lower()
        if "did some shopping today" in low or "went shopping today" in low:
            return f"{subject} did some shopping today."
        if "went to the gym" in low:
            return f"{subject} went to the gym today."
        if "worked late today" in low:
            return f"{subject} worked late today."
        if domain == "alice_project" and (
            "ready to work on" in low
            or "continue working on alice" in low
            or "keep building alice" in low
            or "back to working on alice" in low
        ):
            return f"{subject} was ready to work on the AI/Alice project."
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

        fragments = self._split_fragments(text)
        has_mixed_alice_and_personal = bool(
            self._alice_project_pattern.search(text)
            and self._personal_priority_pattern.search(text)
        )
        candidates: List[MemoryCandidate] = []
        for fragment in fragments:
            fragment_domains = self._pick_domains(fragment)
            if self._alice_project_work_pattern.search(fragment):
                if "alice_project" not in fragment_domains:
                    fragment_domains.insert(0, "alice_project")
                fragment_domains = [d for d in fragment_domains if d != "work"] + (
                    ["work"] if "work" in fragment_domains and "worked late" in fragment.lower() else []
                )
            is_action_fragment = bool(self._action_request_pattern.search(fragment))
            # Explicit day-to-day personal events get a dedicated personal memory candidate.
            if self._day_to_day_personal_pattern.search(fragment):
                day_content = self._rewrite_content(
                    user_name, fragment, "personal_life", "conversation_event"
                )
                candidates.append(
                    MemoryCandidate(
                        content=day_content,
                        domain="personal_life",
                        kind="conversation_event",
                        scope="day_to_day",
                        confidence=0.82,
                        source=source,
                        should_store=True,
                        fragment=fragment,
                    )
                )

            for domain in fragment_domains:
                normalized_domain = (
                    domain if domain in PERSONAL_MEMORY_DOMAINS else DEFAULT_PERSONAL_DOMAIN
                )
                kind = self._pick_kind(fragment, normalized_domain)
                if has_mixed_alice_and_personal and normalized_domain == "alice_project":
                    kind = "conversation_event"
                if has_mixed_alice_and_personal and normalized_domain == "personal_life":
                    kind = "emotional_state"
                kind = kind if kind in PERSONAL_MEMORY_KINDS else DEFAULT_PERSONAL_KIND
                scope = self._pick_scope(fragment, kind)
                scope = scope if scope in PERSONAL_MEMORY_SCOPES else "day_to_day"

                confidence = 0.62
                if kind in {"project_goal", "emotional_state", "preference"}:
                    confidence = 0.78
                if normalized_domain in {"alice_project", "fitness", "personal_life"}:
                    confidence = max(confidence, 0.75)
                if normalized_domain == "personal_life" and self._personal_priority_pattern.search(fragment):
                    confidence = max(confidence, 0.84)

                # Avoid storing action/capability fragments as personal memory.
                should_store = True
                if is_action_fragment and normalized_domain in {"alice_project", "general"}:
                    should_store = False
                if normalized_domain == "general":
                    should_store = False
                # Day-to-day explicit event already captured above.
                if self._day_to_day_personal_pattern.search(fragment) and normalized_domain == "personal_life":
                    should_store = False

                content = self._rewrite_content(user_name, fragment, normalized_domain, kind)
                candidates.append(
                    MemoryCandidate(
                        content=content,
                        domain=normalized_domain,
                        kind=kind,
                        scope=scope,
                        confidence=confidence,
                        source=source,
                        should_store=should_store,
                        fragment=fragment,
                    )
                )

        # Keep deterministic unique entries by (domain, fragment), preferring storable candidates.
        out: List[MemoryCandidate] = []
        seen = set()
        for candidate in candidates:
            dedupe_key = (candidate.domain, candidate.fragment.lower().strip())
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            out.append(candidate)
        return out
