"""Tests for the pre-NLP context resolver layer."""

from ai.context_resolver import ContextResolver


def test_context_resolver_rewrites_list_it_with_last_subject():
    state = {
        "current_topic": "internal codebase",
        "last_subject": "internal codebase",
        "last_intent": "codebase:list",
        "active_goal": "inspect code",
        "referenced_entities": ["internal codebase"],
        "last_entities": {"subject": "internal codebase"},
    }

    resolver = ContextResolver()
    result = resolver.resolve("list it to me", state)

    assert result.needs_clarification is False
    assert "internal codebase" in result.rewritten_input.lower()
    assert result.resolved_bindings.get("it") == "internal codebase"


def test_context_resolver_requests_clarification_without_reference_memory():
    state = {
        "current_topic": "",
        "last_subject": "",
        "last_intent": "",
        "active_goal": "",
        "referenced_entities": [],
        "last_entities": {},
    }
    resolver = ContextResolver()

    result = resolver.resolve("list it", state)

    assert result.needs_clarification is True
    assert "it" in result.unresolved_pronouns
