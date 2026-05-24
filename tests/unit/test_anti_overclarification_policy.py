"""Unit tests for anti-overclarification policy."""

from ai.runtime.anti_overclarification_policy import should_answer_instead_of_clarify


def test_conversation_question_always_answers():
    assert should_answer_instead_of_clarify("what do you think?", "conversation:question")


def test_conversation_general_always_answers():
    assert should_answer_instead_of_clarify("tell me more", "conversation:general")


def test_conversation_ack_always_answers():
    assert should_answer_instead_of_clarify("got it", "conversation:ack")


def test_short_inputs_with_objective_answer():
    state = {"active_objective": "build the routing refactor"}
    assert should_answer_instead_of_clarify("next", "code:request", operator_state=state)
    assert should_answer_instead_of_clarify("and?", "code:request", operator_state=state)


def test_short_inputs_without_objective_and_non_conversation_intent_may_clarify():
    # Short non-conversational inputs without an objective are NOT auto-answered
    result = should_answer_instead_of_clarify("delete", "file_operations:delete")
    assert result is False


def test_risky_inputs_always_clarify():
    assert not should_answer_instead_of_clarify("delete everything", "file_operations:delete")
    assert not should_answer_instead_of_clarify("wipe the disk", "file_operations:delete")


def test_read_file_without_target_clarifies():
    assert not should_answer_instead_of_clarify("read a file", "file_operations:read")


def test_which_question_always_clarifies():
    assert not should_answer_instead_of_clarify("which one?", "conversation:question")
