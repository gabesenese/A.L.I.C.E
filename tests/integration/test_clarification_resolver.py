from ai.core.clarification_resolver import ClarificationResolver


def test_route_choice_slots_are_not_consumed_anymore():
    resolver = ClarificationResolver()
    resolution = resolver.resolve(
        user_input="quick search",
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
            "allowed_values": [
                "intent_routing",
                "entity_extraction",
                "embeddings",
                "conversation_flow",
            ],
        },
    )

    assert resolution.consumed is True
    assert resolution.selected_branch == "conversation_flow"
    assert "nlp algorithms" in resolution.reconstructed_input.lower()
    assert "conversation flow" in resolution.reconstructed_input.lower()
    assert resolution.reconstructed_intent == "conversation:question"
