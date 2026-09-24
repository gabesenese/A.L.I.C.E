"""In conversation she can do what she offers to do.

Only read tools reached an ordinary turn, and there was no reminder tool at all,
so "want me to remind you?" followed by "yes" could not become a reminder. A
model with nothing to call can only say it did it. Reminders, lists and notes
are his and reversible, so they are offered in conversation; writes that change
code still are not.
"""

from datetime import datetime

from ai.core import tool_catalog as tc
from ai.core.llm_engine import ChatResponse, ToolCall
from ai.core.react_loop import ReactLoop
from ai.planning.reminders import ReminderStore
from ai.plugins.notes_plugin import NotesManager, NotesPlugin
from ai.plugins.plugin_system import PluginManager
from ai.plugins.reminder_plugin import ReminderPlugin


class ScriptedLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.offered = []

    def chat_with_tools(self, messages, tools=None, **kwargs):
        self.offered.append({t["function"]["name"] for t in tools or []})
        return self._responses.pop(0) if self._responses else ChatResponse(content="Done.")


def _call(name, arguments):
    return ChatResponse(content="", tool_calls=[ToolCall(name=name, arguments=arguments)])


def _plugins(tmp_path):
    manager = PluginManager(plugins_dir=str(tmp_path / "none"), use_semantic=False)
    reminders = ReminderPlugin(ReminderStore(tmp_path / "reminders.json"))
    notes = NotesPlugin()
    notes.manager = NotesManager(notes_dir=str(tmp_path / "notes"))
    manager.register_plugin(reminders)
    manager.register_plugin(notes)
    return manager, reminders, notes


def test_conversation_is_offered_the_organiser_but_not_code_writes():
    offered = {s["function"]["name"] for s in tc.build_tool_schemas(max_risk=tc.RISK_PERSONAL)}

    assert {"set_reminder", "check_agenda", "add_to_list", "read_list", "create_note"} <= offered
    assert not {"write_workspace_file", "edit_workspace_file", "run_command"} & offered


def test_yes_to_a_reminder_offer_sets_the_reminder(tmp_path):
    manager, reminders, _ = _plugins(tmp_path)
    llm = ScriptedLLM(
        [_call("set_reminder", {"task": "call the bank", "when": "at 11:59pm"}), ChatResponse(content="Done.")]
    )

    result = ReactLoop(llm, plugin_manager=manager).run("yeah, remind me at 11:59pm")

    assert result.stopped_reason != "approval_required"
    [reminder] = reminders.store.pending()
    assert reminder.text == "call the bank"
    assert (reminder.due_at.hour, reminder.due_at.minute) == (23, 59)
    assert reminder.due_at > datetime.now()
    assert "set_reminder" in llm.offered[0]


def test_adding_to_a_list_takes_either_form_of_its_name(tmp_path):
    manager, _, notes = _plugins(tmp_path)

    tc.execute_tool("add_to_list", {"items": "milk", "list_name": "shopping list"}, plugin_manager=manager)
    added = tc.execute_tool("add_to_list", {"items": "eggs", "list_name": "shopping"}, plugin_manager=manager)
    read = tc.execute_tool("read_list", {"list_name": "shopping"}, plugin_manager=manager)

    assert added.summary == "Added eggs to your shopping list."
    assert read.summary == "On your shopping list: milk and eggs."


def test_an_optional_argument_left_out_is_not_an_error(tmp_path):
    manager, _, _ = _plugins(tmp_path)

    result = tc.execute_tool("check_agenda", {}, plugin_manager=manager)

    assert result.success is True
    assert result.summary == "Nothing on your reminders or notes for today."


def test_a_short_yes_to_her_offer_is_not_small_talk():
    """ "yeah, at 4" has no request verb and reads as small talk, which gets no
    tools. Answering her own offer, it is the request."""
    from types import SimpleNamespace

    from ai.runtime.boundaries import boundary_factory as bf

    history = [{"role": "user", "content": "I need to call the bank later"}]
    llm = SimpleNamespace(conversation_history=history + [{"role": "assistant", "content": "Want me to remind you?"}])
    assert bf._answers_an_offer(llm) is True

    llm.conversation_history[-1]["content"] = "Good luck with it. They open at nine."
    assert bf._answers_an_offer(llm) is False
