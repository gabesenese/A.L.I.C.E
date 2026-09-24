"""The first reply of a session mentions the day he has.

The session briefing only knew Google Calendar events, so with a reminder set
and a note due, "morning" was answered as if the day were empty.
"""

from ai.runtime.alice_contract_factory import build_runtime_boundaries
from ai.runtime.contract_pipeline import ContractPipeline
from tests.integration.test_contract_pipeline import _FakeAlice, _FakePlugins

TODAY = "For today: call mom at 5:00 PM."


class _PluginsWithAnAgenda(_FakePlugins):
    def __init__(self, agenda=TODAY):
        super().__init__()
        self.agenda = agenda

    def execute_for_intent(self, intent, query, entities, context):
        if intent == "reminder:agenda":
            return {"success": True, "response": self.agenda}
        return super().execute_for_intent(intent, query, entities, context)


def _pipeline(agenda=TODAY):
    alice = _FakeAlice()
    alice.plugins = _PluginsWithAnAgenda(agenda)
    return ContractPipeline(build_runtime_boundaries(alice))


def test_the_first_reply_mentions_what_is_on_today():
    pipeline = _pipeline()

    first = pipeline.run_turn(user_input="morning", user_id="u1", turn_number=1)
    second = pipeline.run_turn(user_input="tell me something about rivers", user_id="u1", turn_number=2)

    assert first.response_text.endswith(TODAY)
    assert TODAY not in second.response_text


def test_an_empty_day_is_not_announced():
    result = _pipeline("Nothing on your reminders or notes for today.").run_turn(
        user_input="morning", user_id="u1", turn_number=1
    )

    assert "Nothing on your reminders" not in result.response_text
