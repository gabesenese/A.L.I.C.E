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


def test_context_resolver_preserves_local_ai_antecedent_over_cross_turn_subject():
    state = {
        "current_topic": "nlp",
        "last_subject": "nlp",
        "last_intent": "conversation:clarification_needed",
        "active_goal": "help with ai project",
        "referenced_entities": ["nlp"],
        "last_entities": {"topic": "nlp"},
    }

    resolver = ContextResolver()
    user_text = "my ai is not able to correctly give me some informations or it gets the intent wrong"
    result = resolver.resolve(user_text, state)

    assert result.rewritten_input == user_text
    assert "it" not in result.resolved_bindings


def test_context_resolver_does_not_clarify_temporal_deictic_this_weekend():
    state = {
        "current_topic": "",
        "last_subject": "",
        "last_intent": "",
        "active_goal": "",
        "referenced_entities": [],
        "last_entities": {},
    }
    resolver = ContextResolver()

    result = resolver.resolve("hows the weather for this weekend?", state)

    assert result.needs_clarification is False
    assert "this" not in result.unresolved_pronouns
    assert result.rewritten_input == "hows the weather for this weekend?"


def test_context_resolver_does_not_clarify_weather_it_query():
    state = {
        "current_topic": "",
        "last_subject": "",
        "last_intent": "",
        "active_goal": "",
        "referenced_entities": [],
        "last_entities": {},
    }
    resolver = ContextResolver()

    result = resolver.resolve("is it raining tomorrow?", state)

    assert result.needs_clarification is False
    assert result.rewritten_input == "is it raining tomorrow?"
