from ai.core.clarification_resolver import ClarificationResolver


def test_route_choice_quick_search_reconstructs_parent_request():
    resolver = ClarificationResolver()
    resolution = resolver.resolve(
        user_input="quick search",
        pending_slot={
            "slot_type": "route_choice",
            "parent_request": "give me some nlp algorithms",
            "parent_intent": "conversation:help",
        },
    )

    assert resolution.consumed is True
    assert resolution.route_choice == "quick_search"
    assert resolution.reconstructed_input == "give me some nlp algorithms"
    assert resolution.reconstructed_intent == "search"


def test_route_choice_explanation_reconstructs_parent_intent():
    resolver = ClarificationResolver()
    resolution = resolver.resolve(
        user_input="explanation",
        pending_slot={
            "slot_type": "route_choice",
            "parent_request": "compare stemming and lemmatization",
            "parent_intent": "conversation:help",
        },
    )

    assert resolution.consumed is True
    assert resolution.route_choice == "explanation"
    assert resolution.reconstructed_input == "compare stemming and lemmatization"
    assert resolution.reconstructed_intent == "conversation:help"


def test_route_choice_unrecognized_reply_is_not_consumed():
    resolver = ClarificationResolver()
    resolution = resolver.resolve(
        user_input="maybe",
        pending_slot={
            "slot_type": "route_choice",
            "parent_request": "give me some nlp algorithms",
        },
    )

    assert resolution.consumed is False


def test_topic_branch_reply_reconstructs_parent_goal_with_branch_focus():
    resolver = ClarificationResolver()
    resolution = resolver.resolve(
        user_input="conversation flow first",
        pending_slot={
            "slot_type": "topic_branch",
            "parent_request": "help me with my ai project, i need to know some nlp algorithms",
            "parent_intent": "conversation:question",
            "allowed_values": ["intent_routing", "entity_extraction", "embeddings", "conversation_flow"],
        },
    )

    assert resolution.consumed is True
    assert resolution.selected_branch == "conversation_flow"
    assert "nlp algorithms" in resolution.reconstructed_input.lower()
    assert "conversation flow" in resolution.reconstructed_input.lower()
    assert resolution.reconstructed_intent == "conversation:question"
