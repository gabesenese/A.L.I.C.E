from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class TurnRouteDecision:
    mode: str
    subject: str
    complexity: str
    memory_required: bool
    tool_required: bool
    evidence_required: bool
    background_eligible: bool
    reason: str


def _is_greeting(text: str) -> bool:
    normalized = " ".join(str(text or "").lower().split())
    return normalized in {"hi", "hello", "hey", "hi alice", "hello alice", "hey alice"}


def _is_educational_request(text: str) -> bool:
    low = str(text or "").lower()
    return bool(
        re.search(
            r"\b(what is|explain|teach me|learn|basics|overview|introduction to|how does .+ work)\b",
            low,
        )
    )


def _is_concept_refinement_request(text: str, has_active_thread: bool) -> bool:
    low = str(text or "").lower()
    strong_concept_markers = (
        "proactive",
        "not assistant",
        "not chatbot",
        "always running",
        "background monitoring",
        "detect changes",
        "suggest actions",
        "agentic ai",
        "local-first",
        "alice-ollama",
    )
    if "assistant or chatbot" in low and ("dont want" in low or "don't want" in low):
        return True
    if bool(re.search(r"\b(not|dont want|don't want).{0,20}\b(chatbot|assistant)\b", low)):
        return True
    if any(marker in low for marker in strong_concept_markers):
        return True
    if has_active_thread and "ai companion" in low:
        return True
    if has_active_thread and any(token in low for token in ("something like that", "like that", "that approach", "break this down")):
        return True
    return False


def _is_codebase_work_request(text: str) -> bool:
    low = str(text or "").lower()
    return any(
        phrase in low
        for phrase in (
            "check the repo",
            "check the codebase",
            "inspect the project",
            "codebase",
            "read file",
            "analyze file",
            "open file",
            "list files",
            "codex input",
        )
    )


def _is_operator_work_request(text: str) -> bool:
    low = str(text or "").lower()
    return any(
        phrase in low
        for phrase in (
            "implement this in alice",
            "work on alice",
            "ready to work on alice",
            "continue working on alice",
            "improve alice",
        )
    )


def _subject_from_text(text: str) -> str:
    low = str(text or "").lower()
    project_terms = ("alice", "codebase", "repo", "runtime", "project", "agentic")
    personal_terms = ("i ", "my ", "me ", "today", "family", "friend", "workout")
    if any(token in low for token in project_terms) and any(token in low for token in personal_terms):
        return "mixed"
    if any(token in low for token in project_terms):
        return "project"
    if any(token in low for token in ("weather", "news", "market", "machine learning", "math", "history")):
        return "external"
    if any(token in low for token in personal_terms):
        return "personal"
    return "external"


def _complexity_from_text(text: str) -> str:
    low = str(text or "").lower()
    token_count = len(re.findall(r"\b[a-z0-9']+\b", low))
    if _is_codebase_work_request(low) or "architecture" in low or "implement" in low:
        return "high"
    if token_count >= 18:
        return "medium"
    return "low"


def route_turn(
    user_input: str,
    *,
    current_intent: str = "",
    current_route: str = "",
    active_concept_thread: Any = None,
) -> TurnRouteDecision:
    text = str(user_input or "")
    low = text.lower()
    has_active_thread = bool(active_concept_thread and getattr(active_concept_thread, "topic", ""))

    if _is_greeting(low):
        return TurnRouteDecision(
            mode="greeting",
            subject="external",
            complexity="low",
            memory_required=False,
            tool_required=False,
            evidence_required=False,
            background_eligible=False,
            reason="greeting_detected",
        )

    if str(current_route or "").lower() in {"tool", "plugin"}:
        return TurnRouteDecision(
            mode="tool_result",
            subject=_subject_from_text(low),
            complexity=_complexity_from_text(low),
            memory_required=True,
            tool_required=True,
            evidence_required=False,
            background_eligible=False,
            reason="tool_route_active",
        )

    if str(current_route or "").lower() == "clarify" or "clarification" in str(current_intent or "").lower():
        return TurnRouteDecision(
            mode="clarification",
            subject=_subject_from_text(low),
            complexity="low",
            memory_required=False,
            tool_required=False,
            evidence_required=False,
            background_eligible=False,
            reason="clarification_route",
        )

    if _is_codebase_work_request(low):
        return TurnRouteDecision(
            mode="codebase_work",
            subject="project",
            complexity="high",
            memory_required=True,
            tool_required=True,
            evidence_required=True,
            background_eligible=False,
            reason="codebase_request_detected",
        )

    if _is_operator_work_request(low):
        return TurnRouteDecision(
            mode="operator_work",
            subject="project",
            complexity="high",
            memory_required=True,
            tool_required=False,
            evidence_required=True,
            background_eligible=False,
            reason="operator_work_request_detected",
        )

    if _is_educational_request(low):
        if not _is_concept_refinement_request(low, has_active_thread):
            subject = _subject_from_text(low)
            return TurnRouteDecision(
                mode="educational_explain",
                subject=subject,
                complexity="medium",
                memory_required=(subject != "external"),
                tool_required=False,
                evidence_required=False,
                background_eligible=False,
                reason="educational_request_detected",
            )

    if _is_concept_refinement_request(low, has_active_thread):
        return TurnRouteDecision(
            mode="concept_refinement",
            subject="project",
            complexity="medium",
            memory_required=True,
            tool_required=False,
            evidence_required=False,
            background_eligible=False,
            reason="concept_refinement_detected",
        )

    return TurnRouteDecision(
        mode="companion_chat",
        subject=_subject_from_text(low),
        complexity=_complexity_from_text(low),
        memory_required=True,
        tool_required=False,
        evidence_required=False,
        background_eligible=False,
        reason="default_companion_chat",
    )
