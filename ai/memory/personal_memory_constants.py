"""Structured personal-memory taxonomy constants."""

from __future__ import annotations

PERSONAL_MEMORY_DOMAINS: set[str] = {
    "personal_life",
    "alice_project",
    "work",
    "finance",
    "fitness",
    "relationships",
    "health",
    "preferences",
    "general",
}

PERSONAL_MEMORY_KINDS: set[str] = {
    "personal_fact",
    "preference",
    "project_goal",
    "emotional_state",
    "routine",
    "relationship_context",
    "daily_summary",
    "long_term_profile",
    "conversation_event",
}

PERSONAL_MEMORY_SCOPES: set[str] = {"day_to_day", "long_term"}

DEFAULT_PERSONAL_DOMAIN = "general"
DEFAULT_PERSONAL_KIND = "conversation_event"
DEFAULT_PERSONAL_SCOPE = "day_to_day"
