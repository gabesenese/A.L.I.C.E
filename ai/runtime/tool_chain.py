"""
Tool chaining engine for A.L.I.C.E — Jarvis-style multi-tool execution.

Detects compound queries (e.g. "schedule and emails", "system status and goals")
and executes multiple plugins in a single turn, merging their responses into one
coherent output. All logic is pure-Python with no LLM calls.

Public API:
    detect_chain(query, primary_intent) -> List[str]  -- returns intent list to chain
    execute_chain(intents, query, entities, context, plugin_manager) -> str | None
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

# ---------------------------------------------------------------------------
# Domain keyword maps — map surface words to plugin intent prefixes
# ---------------------------------------------------------------------------
_DOMAIN_KEYWORDS: Dict[str, Set[str]] = {
    "calendar": {"schedule", "calendar", "meeting", "meetings", "event", "events", "appointment"},
    "email":    {"email", "emails", "inbox", "mail", "message", "messages", "unread"},
    "system":   {"system", "cpu", "memory", "ram", "disk", "processes", "ports", "network", "status"},
    "weather":  {"weather", "temperature", "forecast", "rain", "sunny", "wind", "humidity"},
    "notes":    {"note", "notes", "jot", "write down"},
    "search":   {"search", "look up", "google", "find", "what is", "who is"},
    "goals":    {"goals", "goal", "objective", "objectives", "priorities", "priority"},
    "tasks":    {"task", "tasks", "todo", "to-do", "open threads", "pending"},
}

# Connective patterns that suggest the user wants multiple things
_COMPOUND_PATTERNS = re.compile(
    r"\b(and|also|plus|as well|along with|together with|both|while you.?re at it|"
    r"what about|show me|give me|tell me|any|check)\b",
    re.IGNORECASE,
)

# Map detected domain → canonical intent to execute
_DOMAIN_TO_INTENT: Dict[str, str] = {
    "calendar": "calendar:events",
    "email":    "email:check",
    "system":   "system:status",
    "weather":  "weather:current",
    "notes":    "notes:list",
    "goals":    "goals:list",
    "tasks":    "tasks:list",
}


def _detect_domains(query: str) -> List[str]:
    """Return ordered list of domain names found in the query."""
    q = query.lower()
    found: List[str] = []
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                found.append(domain)
                break
    return found


def detect_chain(query: str, primary_intent: str) -> List[str]:
    """Determine if the query warrants chaining multiple tools.

    Returns a list of intents to execute. If only one tool is needed,
    returns an empty list (let the normal pipeline handle it).
    """
    domains = _detect_domains(query)
    if len(domains) < 2:
        return []

    # Must have a connective word or explicit listing structure
    has_connector = bool(_COMPOUND_PATTERNS.search(query))
    has_comma = "," in query
    if not has_connector and not has_comma:
        return []

    # Map domains → intents, preserving detection order, skip duplicates
    seen: Set[str] = set()
    intents: List[str] = []

    # Put the primary intent's domain first if present
    primary_domain = str(primary_intent or "").split(":")[0]
    if primary_domain in _DOMAIN_TO_INTENT:
        intent = _DOMAIN_TO_INTENT[primary_domain]
        if intent not in seen:
            seen.add(intent)
            intents.append(intent)

    for domain in domains:
        intent = _DOMAIN_TO_INTENT.get(domain)
        if intent and intent not in seen:
            seen.add(intent)
            intents.append(intent)

    return intents if len(intents) >= 2 else []


def execute_chain(
    intents: List[str],
    query: str,
    entities: Dict[str, Any],
    context: Dict[str, Any],
    plugin_manager: Any,
) -> Optional[str]:
    """Execute multiple tool intents and merge their responses.

    Returns a merged response string, or None if all tools failed.
    """
    if not intents:
        return None

    results: List[str] = []
    failed: List[str] = []

    for intent in intents:
        try:
            # Find a plugin that can handle this intent
            plugin = _find_plugin(intent, query, entities, plugin_manager)
            if plugin is None:
                failed.append(intent)
                continue

            result = plugin.execute(
                intent=intent,
                query=query,
                entities=entities,
                context=context,
            )

            if not isinstance(result, dict):
                failed.append(intent)
                continue

            if result.get("success") and result.get("response"):
                results.append(str(result["response"]).strip())
            else:
                failed.append(intent)
        except Exception:
            failed.append(intent)

    if not results:
        return None

    merged = "\n\n".join(results)
    if failed:
        failed_labels = ", ".join(f.split(":")[0] for f in failed)
        merged += f"\n\n(Could not retrieve: {failed_labels})"
    return merged


def _find_plugin(
    intent: str,
    query: str,
    entities: Dict[str, Any],
    plugin_manager: Any,
) -> Optional[Any]:
    """Find the first registered plugin that can handle this intent."""
    try:
        # plugin_manager.plugins is a dict[name → PluginInterface]
        plugins_dict = getattr(plugin_manager, "plugins", {})
        for instance in plugins_dict.values():
            if not getattr(instance, "enabled", True):
                continue
            if hasattr(instance, "can_handle") and hasattr(instance, "execute"):
                try:
                    if instance.can_handle(intent, entities, query):
                        return instance
                except Exception:
                    pass
    except Exception:
        pass
    return None
