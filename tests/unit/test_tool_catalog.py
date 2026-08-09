"""The tool surface the model selects from, and the guards around dispatching it."""

import pytest

from ai.core import tool_catalog as tc


def test_every_spec_produces_a_wellformed_ollama_schema():
    for schema in tc.build_tool_schemas():
        assert schema["type"] == "function"
        function = schema["function"]
        assert function["name"]
        assert function["description"].strip()
        assert function["parameters"]["type"] == "object"
        assert isinstance(function["parameters"]["properties"], dict)
        for required in function["parameters"].get("required", []):
            assert required in function["parameters"]["properties"]


def test_schema_names_are_unique():
    names = [s["function"]["name"] for s in tc.build_tool_schemas()]
    assert len(names) == len(set(names))


def test_risk_filter_hides_write_tools_from_a_read_only_caller():
    read_only = {s["function"]["name"] for s in tc.build_tool_schemas(max_risk=tc.RISK_READ)}
    everything = {s["function"]["name"] for s in tc.build_tool_schemas(max_risk=tc.RISK_OUTWARD)}
    assert "create_note" in everything
    assert "create_note" not in read_only
    assert "list_workspace_files" in read_only


def test_named_subset_narrows_the_surface():
    schemas = tc.build_tool_schemas(names=["search_workspace"])
    assert [s["function"]["name"] for s in schemas] == ["search_workspace"]


def test_sanitize_drops_nulls_empties_and_unknown_keys():
    spec = tc.get_spec("search_workspace")
    cleaned = tc.sanitize_arguments(spec, {"query": "route", "subdirectory": None, "bogus": 1, "limit": ""})
    assert cleaned == {"query": "route"}


def test_unknown_tool_is_refused_not_executed():
    result = tc.execute_tool("rm_rf_everything", {})
    assert result.success is False
    assert "Unknown tool" in result.error


def test_missing_required_argument_is_reported():
    result = tc.execute_tool("read_workspace_file", {})
    assert result.success is False
    assert "path" in result.error


@pytest.mark.parametrize("escape", ["../../../../Windows/System32/drivers/etc/hosts", "/etc/passwd", "../../.ssh/id_rsa"])
def test_reads_outside_the_workspace_are_blocked(escape):
    result = tc.execute_tool("read_workspace_file", {"path": escape})
    assert result.success is False


def test_list_workspace_files_returns_real_project_files():
    result = tc.execute_tool("list_workspace_files", {})
    assert result.success is True
    files = result.data["files"]
    assert "ai/core/tool_catalog.py" in files
    assert all(".venv" not in f and "__pycache__" not in f for f in files)


def test_list_workspace_files_scopes_to_a_subdirectory():
    result = tc.execute_tool("list_workspace_files", {"subdirectory": "ai/runtime"})
    assert result.success is True
    assert result.data["files"]
    assert all(f.startswith("ai/runtime/") for f in result.data["files"])


def test_read_workspace_file_returns_real_content():
    result = tc.execute_tool("read_workspace_file", {"path": "ai/core/tool_catalog.py"})
    assert result.success is True
    assert "class ToolSpec" in result.data["content"]
    assert result.data["line_count"] > 100


def test_read_workspace_file_truncates_and_says_so():
    result = tc.execute_tool("read_workspace_file", {"path": "ai/core/tool_catalog.py", "max_chars": 300})
    assert result.success is True
    assert len(result.data["content"]) == 300
    assert result.data["truncated"] is True


def test_search_workspace_finds_a_known_symbol():
    result = tc.execute_tool("search_workspace", {"query": "def build_tool_schemas"})
    assert result.success is True
    assert result.data["match_count"] >= 1
    assert any(m["path"] == "ai/core/tool_catalog.py" for m in result.data["matches"])


def test_search_workspace_rejects_an_empty_query():
    result = tc.execute_tool("search_workspace", {"query": "   "})
    assert result.success is False


def test_plugin_backed_tool_reports_when_no_plugin_manager_is_available():
    result = tc.execute_tool("get_system_status", {})
    assert result.success is False
    assert "plugin manager" in result.error.lower()


def test_plugin_backed_tool_dispatches_through_execute_for_intent():
    calls = {}

    class FakePluginManager:
        def execute_for_intent(self, intent, query, entities, context):
            calls.update(intent=intent, query=query, entities=entities)
            return {"success": True, "response": "CPU 12%"}

    result = tc.execute_tool("get_current_weather", {"location": "Toronto"}, plugin_manager=FakePluginManager())
    assert result.success is True
    assert result.summary == "CPU 12%"
    assert calls["intent"] == "weather:current"
    assert calls["entities"] == {"location": "Toronto"}


def test_plugin_failure_surfaces_as_a_failed_execution():
    class FailingPluginManager:
        def execute_for_intent(self, intent, query, entities, context):
            raise RuntimeError("plugin exploded")

    result = tc.execute_tool("list_notes", {}, plugin_manager=FailingPluginManager())
    assert result.success is False
    assert "plugin exploded" in result.error
