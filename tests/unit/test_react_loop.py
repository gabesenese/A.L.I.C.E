"""The reason/act/observe loop, driven by a scripted model so it runs offline."""

from ai.core import tool_catalog as tc
from ai.core.llm_engine import ChatResponse, ToolCall
from ai.core.react_loop import ReactLoop


class ScriptedLLM:
    """Returns a queued response per call and records what it was sent."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def chat_with_tools(self, messages, tools=None, **kwargs):
        self.calls.append({"messages": list(messages), "tools": tools})
        if not self._responses:
            return ChatResponse(content="done")
        return self._responses.pop(0)


def tool_response(name, arguments):
    return ChatResponse(
        content="",
        tool_calls=[ToolCall(name=name, arguments=arguments)],
        raw={"message": {"tool_calls": [{"function": {"name": name, "arguments": arguments}}]}},
    )


def test_answers_directly_when_no_tool_is_needed():
    llm = ScriptedLLM([ChatResponse(content="Nothing to look up.")])
    result = ReactLoop(llm).run("how are you")
    assert result.used_tools is False
    assert result.answer == "Nothing to look up."
    assert result.stopped_reason == "answered"


def test_executes_a_tool_and_answers_from_the_observation():
    llm = ScriptedLLM(
        [
            tool_response("search_workspace", {"query": "def build_tool_schemas"}),
            ChatResponse(content="It is defined in ai/core/tool_catalog.py."),
        ]
    )
    result = ReactLoop(llm).run("where is build_tool_schemas defined")

    assert result.used_tools is True
    assert [s.tool for s in result.steps] == ["search_workspace"]
    assert result.steps[0].success is True
    assert result.answer == "It is defined in ai/core/tool_catalog.py."


def test_observation_is_fed_back_to_the_model():
    llm = ScriptedLLM(
        [
            tool_response("search_workspace", {"query": "def build_tool_schemas"}),
            ChatResponse(content="found it"),
        ]
    )
    ReactLoop(llm).run("where is build_tool_schemas")

    second_call_roles = [m["role"] for m in llm.calls[1]["messages"]]
    assert "tool" in second_call_roles
    tool_message = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"][0]
    assert "tool_catalog.py" in tool_message["content"]


def test_chains_two_tools_across_steps():
    llm = ScriptedLLM(
        [
            tool_response("list_workspace_files", {"subdirectory": "ai/core"}),
            tool_response("read_workspace_file", {"path": "ai/core/tool_catalog.py"}),
            ChatResponse(content="It defines the tool surface."),
        ]
    )
    result = ReactLoop(llm).run("summarize the tool catalog")

    assert [s.tool for s in result.steps] == ["list_workspace_files", "read_workspace_file"]
    assert result.answer == "It defines the tool surface."


def test_step_budget_stops_a_model_that_never_answers():
    llm = ScriptedLLM([tool_response("list_workspace_files", {}) for _ in range(20)])
    result = ReactLoop(llm, max_steps=3).run("keep going forever")
    assert result.stopped_reason == "step_budget_exhausted"
    assert len(result.steps) == 3


def test_repeated_identical_call_is_told_to_use_the_earlier_result():
    llm = ScriptedLLM(
        [
            tool_response("list_workspace_files", {}),
            tool_response("list_workspace_files", {}),
            ChatResponse(content="ok"),
        ]
    )
    ReactLoop(llm).run("list files twice")
    tool_messages = [m["content"] for m in llm.calls[2]["messages"] if m["role"] == "tool"]
    assert any("already called this tool" in m for m in tool_messages)


def test_write_tier_tool_stops_the_loop_and_requests_approval():
    llm = ScriptedLLM([tool_response("create_note", {"title": "buy milk"})])
    result = ReactLoop(llm, allow_write_tools=False).run("make me a note")

    assert result.stopped_reason == "approval_required"
    assert result.pending_approval["tool"] == "create_note"
    assert result.pending_approval["arguments"] == {"title": "buy milk"}
    assert result.steps == []


def test_unknown_tool_is_reported_back_rather_than_executed():
    llm = ScriptedLLM([tool_response("delete_everything", {}), ChatResponse(content="cannot do that")])
    result = ReactLoop(llm).run("delete everything")

    assert [s.tool for s in result.steps] == ["delete_everything"]
    assert result.steps[0].success is False
    tool_messages = [m["content"] for m in llm.calls[1]["messages"] if m["role"] == "tool"]
    assert any("unknown tool" in m for m in tool_messages)


def test_model_failure_stops_cleanly_instead_of_raising():
    class BrokenLLM:
        def chat_with_tools(self, messages, tools=None, **kwargs):
            raise RuntimeError("ollama down")

    result = ReactLoop(BrokenLLM()).run("anything")
    assert result.stopped_reason == "model_error"
    assert result.answer == ""


def test_tool_schemas_are_actually_passed_to_the_model():
    llm = ScriptedLLM([ChatResponse(content="hi")])
    ReactLoop(llm).run("hello")
    sent = llm.calls[0]["tools"]
    assert sent, "the loop must send tool definitions"
    assert {t["function"]["name"] for t in sent} >= {"list_workspace_files", "read_workspace_file"}


def test_write_tools_are_offered_only_when_permitted():
    read_only = {s["function"]["name"] for s in tc.build_tool_schemas(max_risk=tc.RISK_READ)}
    assert "create_note" not in read_only
