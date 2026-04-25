from contextlib import contextmanager
from types import SimpleNamespace

from ai.infrastructure.runtime_flags import is_enabled
from ai.introspection.system_state_api import SystemStateAPI
from ai.plugins.file_operations_plugin import FileOperationsPlugin
from ai.plugins.plugin_system import PluginInterface, PluginManager


def test_rich_entrypoint_passes_llm_policy(monkeypatch):
    import app.alice as alice_entry

    captured = {}

    class _FakeAlice:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def shutdown(self):
            captured["shutdown"] = True

    class _FakeUI:
        def __init__(self, user_name):
            self.user_name = user_name

        def show_loading(self, message):
            pass

        def clear(self):
            pass

        def show_welcome(self):
            pass

        def print_info(self, message):
            pass

        def get_input(self):
            return "exit"

        def show_goodbye(self):
            pass

        def print_error(self, message):
            raise AssertionError(message)

        def print_user_input(self, text):
            pass

        def print_assistant_response(self, response):
            pass

        @contextmanager
        def thinking_spinner(self):
            yield

    monkeypatch.setattr(alice_entry, "ALICE", _FakeAlice)
    monkeypatch.setattr("ui.rich_terminal.RichTerminalUI", _FakeUI)

    alice_entry.start_alice_rich(llm_policy="strict")

    assert captured["llm_policy"] == "strict"
    assert captured["shutdown"] is True


class _FakeSemanticClassifier:
    def get_plugin_action(self, query, threshold=0.45):
        return {"plugin": "calendar", "action": "view", "confidence": 0.91}


class _FakeCalendarPlugin(PluginInterface):
    def __init__(self):
        super().__init__()
        self.name = "CalendarPlugin"

    def initialize(self) -> bool:
        return True

    def can_handle(self, intent, entities, query=None) -> bool:
        return True

    def execute(self, intent, query, entities, context):
        return {"success": True, "response": "calendar ok"}

    def shutdown(self) -> None:
        pass


def test_semantic_plugin_mapping_matches_registered_calendar_name():
    manager = PluginManager(use_semantic=False)
    manager.register_plugin(_FakeCalendarPlugin())
    manager.use_semantic = True
    manager.intent_classifier = _FakeSemanticClassifier()

    result = manager.execute_for_intent("calendar", "show my calendar", {}, {})

    assert result["success"] is True
    assert result["plugin"] == "CalendarPlugin"
    assert result["response"] == "calendar ok"


def test_file_operations_rejects_sibling_prefix_escape(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    sibling = tmp_path / "base_evil"
    sibling.mkdir()
    inside = base / "note.txt"
    outside = sibling / "note.txt"
    inside.write_text("ok", encoding="utf-8")
    outside.write_text("no", encoding="utf-8")

    plugin = FileOperationsPlugin()
    plugin.safe_base_dir = str(base.resolve())

    assert plugin._is_safe_path(str(inside)) is True
    assert plugin._is_safe_path(str(outside)) is False


def test_system_state_api_reports_actual_runtime_attribute_names():
    class _GoalSystem:
        def get_active_goals(self):
            return [SimpleNamespace(title="Build desktop mode", status="active")]

    alice = SimpleNamespace(
        nlp=object(),
        router=None,
        reasoning_engine=object(),
        learning_engine=object(),
        goal_system=_GoalSystem(),
        plugins=SimpleNamespace(plugins={"CalendarPlugin": object()}),
    )

    api = SystemStateAPI(alice)

    processors = api.get_processor_state()
    plugins = api.get_plugin_state()
    goals = api.get_active_goals()

    assert processors["reasoning_engine"] == "active"
    assert processors["learning_engine"] == "active"
    assert plugins["plugins_loaded"] is True
    assert plugins["plugin_count"] == 1
    assert plugins["available_plugins"] == ["CalendarPlugin"]
    assert goals == ["Build desktop mode (active)"]


def test_contract_pipeline_remains_enabled_by_default():
    assert is_enabled("contract_pipeline") is True
