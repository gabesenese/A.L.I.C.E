"""Approval is for what would actually be done, not for words in what is said.

"remind me to post the letter" and "add kill the weeds to my todo list" were
stopped for approval over "post" and "kill", as was "remove eggs from the
shopping list". Notes ask before removing a note themselves.
"""

import pytest

from ai.runtime.companion_runtime import CompanionPolicyEngine


@pytest.mark.parametrize(
    "text, intent",
    [
        ("remind me to post the letter tomorrow at 9", "reminder:set"),
        ("remind me to delete the old backups at 8pm", "reminder:set"),
        ("add kill the weeds to my todo list", "notes:append"),
        ("remove eggs from the shopping list", "notes:list"),
        ("remind me to wipe down the counters at 6", "reminder:set"),
        ("forget my favorite color", "memory:delete"),
    ],
)
def test_what_a_reminder_or_note_says_is_not_an_action(text, intent):
    assert CompanionPolicyEngine().requires_approval(user_input=text, intent=intent) == (False, "")


def test_a_real_destructive_command_still_asks():
    needed, _ = CompanionPolicyEngine().requires_approval(user_input="force push to main", intent="system:git")
    assert needed is True


def test_the_question_names_what_would_be_deleted():
    from types import SimpleNamespace

    from ai.runtime.companion_runtime import CompanionRuntimeLoop

    decision = SimpleNamespace(intent="file_operations:delete", metadata={"resolved_input": "delete the old logs"})

    assert CompanionRuntimeLoop._dry_run_preview(decision=decision) == "permanently delete the old logs"
