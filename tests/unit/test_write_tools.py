"""The write tier: real file mutation, contained to the workspace."""

import pytest

from ai.core import tool_catalog as tc
from ai.core.llm_engine import ChatResponse, ToolCall
from ai.core.react_loop import ReactLoop

SCRATCH = "data/test_scratch"


@pytest.fixture
def scratch_file(tmp_path, monkeypatch):
    """Point the workspace root at a temp directory so tests never touch the repo."""
    monkeypatch.setenv("ALICE_PROJECT_ROOT", str(tmp_path))
    (tmp_path / "pkg").mkdir()
    target = tmp_path / "pkg" / "sample.py"
    target.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    return target


def test_write_creates_a_new_file(scratch_file, tmp_path):
    result = tc.execute_tool("write_workspace_file", {"path": "pkg/new.py", "content": "x = 1\n"})
    assert result.success is True
    assert result.data["created"] is True
    assert (tmp_path / "pkg" / "new.py").read_text(encoding="utf-8") == "x = 1\n"


def test_write_refuses_to_clobber_an_existing_file(scratch_file):
    """Partial content written over a real file silently deletes everything else."""
    original = scratch_file.read_text(encoding="utf-8")
    result = tc.execute_tool("write_workspace_file", {"path": "pkg/sample.py", "content": "y = 2\n"})

    assert result.success is False
    assert "already exists" in result.error
    assert "edit_workspace_file" in result.error
    assert scratch_file.read_text(encoding="utf-8") == original


def test_write_replaces_an_existing_file_when_overwrite_is_explicit(scratch_file):
    result = tc.execute_tool(
        "write_workspace_file",
        {"path": "pkg/sample.py", "content": "y = 2\n", "overwrite": True},
    )
    assert result.success is True
    assert result.data["created"] is False
    assert scratch_file.read_text(encoding="utf-8") == "y = 2\n"


def test_write_outside_the_workspace_is_rejected(scratch_file):
    result = tc.execute_tool("write_workspace_file", {"path": "../escaped.py", "content": "nope"})
    assert result.success is False
    assert "outside the workspace" in result.error


def test_edit_replaces_a_unique_block(scratch_file):
    result = tc.execute_tool(
        "edit_workspace_file",
        {"path": "pkg/sample.py", "find": "return a + b", "replace": "return a - b"},
    )
    assert result.success is True
    assert "return a - b" in scratch_file.read_text(encoding="utf-8")


def test_edit_refuses_ambiguous_text_rather_than_guessing(scratch_file):
    scratch_file.write_text("x = 1\nx = 1\n", encoding="utf-8")
    result = tc.execute_tool("edit_workspace_file", {"path": "pkg/sample.py", "find": "x = 1", "replace": "x = 2"})
    assert result.success is False
    assert "appears 2 times" in result.error
    assert scratch_file.read_text(encoding="utf-8") == "x = 1\nx = 1\n"


def test_edit_reports_when_the_text_is_absent(scratch_file):
    result = tc.execute_tool("edit_workspace_file", {"path": "pkg/sample.py", "find": "nonexistent", "replace": "x"})
    assert result.success is False
    assert "not found" in result.error


def test_edit_requires_an_existing_file(scratch_file):
    result = tc.execute_tool("edit_workspace_file", {"path": "pkg/ghost.py", "find": "a", "replace": "b"})
    assert result.success is False
    assert "No such file" in result.error


def test_run_command_captures_output_and_exit_code(scratch_file):
    result = tc.execute_tool("run_command", {"command": "python -c \"print('hello')\""})
    assert result.success is True
    assert "hello" in result.data["stdout"]
    assert result.data["exit_code"] == 0


def test_run_command_reports_a_failing_exit_code(scratch_file):
    result = tc.execute_tool("run_command", {"command": "python -c \"import sys; sys.exit(3)\""})
    assert result.success is False
    assert result.data["exit_code"] == 3


def _tool_response(name, arguments):
    return ChatResponse(
        content="",
        tool_calls=[ToolCall(name=name, arguments=arguments)],
        raw={"message": {"tool_calls": [{"function": {"name": name, "arguments": arguments}}]}},
    )


class ScriptedLLM:
    def __init__(self, responses):
        self._responses = list(responses)

    def chat_with_tools(self, messages, tools=None, **kwargs):
        return self._responses.pop(0) if self._responses else ChatResponse(content="done")


def test_loop_refuses_a_destructive_command_without_asking(scratch_file):
    llm = ScriptedLLM([_tool_response("run_command", {"command": "rm -rf /"})])
    result = ReactLoop(llm, allow_write_tools=True, checkpoint_writes=False).run("clean up")

    assert result.stopped_reason == "refused"
    assert result.refused["reason"] == "destructive_command"
    assert result.steps == []


def test_loop_stops_for_approval_before_writing_outside_the_workspace(scratch_file):
    llm = ScriptedLLM([_tool_response("write_workspace_file", {"path": "../outside.txt", "content": "x"})])
    result = ReactLoop(llm, allow_write_tools=True, checkpoint_writes=False).run("write that file")

    assert result.stopped_reason == "approval_required"
    assert result.pending_approval["reason"] == "writes_outside_workspace"


def test_loop_performs_a_workspace_write_unattended(scratch_file, tmp_path):
    llm = ScriptedLLM(
        [
            _tool_response("write_workspace_file", {"path": "pkg/generated.py", "content": "z = 3\n"}),
            ChatResponse(content="Written."),
        ]
    )
    result = ReactLoop(llm, allow_write_tools=True, checkpoint_writes=False).run("create that file")

    assert result.stopped_reason == "answered"
    assert [s.tool for s in result.steps] == ["write_workspace_file"]
    assert result.steps[0].success is True
    assert (tmp_path / "pkg" / "generated.py").read_text(encoding="utf-8") == "z = 3\n"


def test_loop_stops_for_approval_before_overwriting_an_existing_file(scratch_file):
    """The model can ask for an overwrite. It still does not get one unattended."""
    original = scratch_file.read_text(encoding="utf-8")
    llm = ScriptedLLM(
        [_tool_response("write_workspace_file", {"path": "pkg/sample.py", "content": "gone", "overwrite": True})]
    )
    result = ReactLoop(llm, allow_write_tools=True, checkpoint_writes=False).run("rewrite that file")

    assert result.stopped_reason == "approval_required"
    assert result.pending_approval["reason"] == "overwrites_existing_file"
    assert scratch_file.read_text(encoding="utf-8") == original


def test_creating_a_brand_new_file_still_runs_unattended(scratch_file, tmp_path):
    llm = ScriptedLLM(
        [
            _tool_response("write_workspace_file", {"path": "pkg/fresh.py", "content": "a = 1\n", "overwrite": True}),
            ChatResponse(content="Created."),
        ]
    )
    result = ReactLoop(llm, allow_write_tools=True, checkpoint_writes=False).run("create it")
    assert result.stopped_reason == "answered"
    assert (tmp_path / "pkg" / "fresh.py").exists()


def test_write_tools_are_withheld_when_the_loop_is_read_only(scratch_file):
    llm = ScriptedLLM([_tool_response("write_workspace_file", {"path": "pkg/x.py", "content": "1"})])
    result = ReactLoop(llm, allow_write_tools=False, checkpoint_writes=False).run("write something")
    assert result.stopped_reason == "approval_required"
