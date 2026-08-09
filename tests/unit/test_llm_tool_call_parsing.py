"""Parsing of native tool calls returned by Ollama.

achat previously discarded its `tools` argument outright (`_ = tools`), so no tool
schema ever reached the model and every answer was generated from the prompt alone.
"""

from ai.core.llm_engine import LocalLLMEngine


def parse(message):
    return LocalLLMEngine._parse_tool_calls(message)


def test_parses_a_tool_call_with_dict_arguments():
    calls = parse(
        {
            "tool_calls": [
                {"function": {"name": "search_workspace", "arguments": {"query": "route", "limit": 5}}},
            ]
        }
    )
    assert len(calls) == 1
    assert calls[0].name == "search_workspace"
    assert calls[0].arguments == {"query": "route", "limit": 5}


def test_parses_arguments_delivered_as_a_json_string():
    calls = parse({"tool_calls": [{"function": {"name": "read_workspace_file", "arguments": '{"path": "a.py"}'}}]})
    assert calls[0].arguments == {"path": "a.py"}


def test_drops_null_arguments_the_model_emits_for_optional_fields():
    calls = parse(
        {"tool_calls": [{"function": {"name": "list_workspace_files", "arguments": {"subdirectory": None}}}]}
    )
    assert calls[0].arguments == {}


def test_malformed_argument_json_yields_no_arguments_rather_than_raising():
    calls = parse({"tool_calls": [{"function": {"name": "list_notes", "arguments": "{not json"}}]})
    assert calls[0].name == "list_notes"
    assert calls[0].arguments == {}


def test_tool_call_without_a_name_is_skipped():
    calls = parse({"tool_calls": [{"function": {"arguments": {"a": 1}}}, {"function": {"name": "list_notes"}}]})
    assert [c.name for c in calls] == ["list_notes"]


def test_message_without_tool_calls_returns_empty_list():
    assert parse({"content": "just talking"}) == []
    assert parse({}) == []


def test_multiple_tool_calls_are_all_returned():
    calls = parse(
        {
            "tool_calls": [
                {"function": {"name": "get_system_status", "arguments": {}}},
                {"function": {"name": "list_notes", "arguments": {}}},
            ]
        }
    )
    assert [c.name for c in calls] == ["get_system_status", "list_notes"]
