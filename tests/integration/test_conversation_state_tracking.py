from ai.memory.conversation_state import ConversationStateTracker


def test_learning_question_chain_tracks_topic_and_depth() -> None:
    tracker = ConversationStateTracker(max_chain=8, max_depth=5)

    tracker.update_state(
        user_input="what is polymorphism",
        intent="learning:study_topic",
        entities={"topic": "polymorphism"},
    )
    tracker.update_state(
        user_input="why is it useful",
        intent="conversation:question",
        entities={},
    )
    state = tracker.update_state(
        user_input="can you show example",
        intent="conversation:question",
        entities={},
    )

    assert state["conversation_topic"] == "polymorphism"
    assert state["conversation_goal"] == "learning"
    assert state["depth_level"] >= 3
    assert len(state["question_chain"]) == 3
    assert state["question_chain"][0].lower().startswith("what is polymorphism")


def test_topic_shift_resets_chain_and_depth() -> None:
    tracker = ConversationStateTracker(max_chain=8, max_depth=5)

    tracker.update_state(
        user_input="what is polymorphism",
        intent="learning:study_topic",
        entities={"topic": "polymorphism"},
    )
    tracker.update_state(
        user_input="why is it useful",
        intent="conversation:question",
        entities={},
    )

    state = tracker.update_state(
        user_input="what is docker",
        intent="question:tech",
        entities={},
    )

    assert state["conversation_topic"] == "docker"
    assert state["depth_level"] == 1
    assert len(state["question_chain"]) == 1
    assert state["question_chain"][0].lower().startswith("what is docker")


def test_user_goal_is_extracted_from_prompt() -> None:
    tracker = ConversationStateTracker(max_chain=8, max_depth=5)

    state = tracker.update_state(
        user_input="help me prepare for my oop interview",
        intent="learning:study_topic",
        entities={"topic": "oop"},
    )

    assert state["conversation_goal"] == "learning"
    assert "prepare for my oop interview" in state["user_goal"].lower()


def test_prompt_format_contains_state_fields() -> None:
    tracker = ConversationStateTracker(max_chain=8, max_depth=5)

    tracker.update_state(
        user_input="what is polymorphism",
        intent="learning:study_topic",
        entities={"topic": "polymorphism"},
    )
    tracker.update_state(
        user_input="can you show code example",
        intent="conversation:question",
        entities={},
    )

    context = tracker.format_for_prompt()
    assert "Conversation state:" in context
    assert "topic: polymorphism" in context
    assert "depth_level:" in context
    assert "question_chain:" in context
