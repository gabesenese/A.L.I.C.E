"""Asking permission has to lead somewhere.

The confirm tier used to return a pending approval that nothing stored and nothing
could act on, so "that needs your go-ahead" was a dead end: there was no way to say
yes. This covers the whole round trip, including rollback of a half finished write.
"""

import pytest

from ai.core import tool_catalog as tc
from ai.core.llm_engine import ChatResponse, ToolCall
from ai.core.react_loop import ReactLoop
from ai.runtime import pending_actions
from ai.runtime.boundaries.boundary_factory import (
    _describe_action,
    _is_approval_phrase,
    _is_rejection_phrase,
    _request_approval,
    _resolve_pending_action,
)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("ALICE_PROJECT_ROOT", str(tmp_path))
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "sample.py").write_text("original\n", encoding="utf-8")
    return tmp_path


class Req:
    def __init__(self, user_input):
        self.user_input = user_input
        self.metadata = {"user_id": "approver"}


class FakeAlice:
    plugins = None


def test_request_then_yes_executes_the_action(workspace):
    approval = _request_approval(
        {
            "tool": "write_workspace_file",
            "arguments": {"path": "pkg/sample.py", "content": "replaced\n", "overwrite": True},
            "reason": "overwrites_existing_file",
        },
        user_id="approver",
    )
    assert "pkg/sample.py" in approval.text
    assert approval.metadata["type"] == "approval_requested"
    assert pending_actions.get("approver") is not None

    settled = _resolve_pending_action(FakeAlice(), Req("yes"), "approver")
    assert settled.metadata["type"] == "approved_action_executed"
    assert settled.metadata["success"] is True
    assert (workspace / "pkg" / "sample.py").read_text(encoding="utf-8") == "replaced\n"
    assert pending_actions.get("approver") is None


def test_request_then_no_drops_it_without_acting(workspace):
    _request_approval(
        {
            "tool": "write_workspace_file",
            "arguments": {"path": "pkg/sample.py", "content": "replaced\n", "overwrite": True},
            "reason": "overwrites_existing_file",
        },
        user_id="approver",
    )
    settled = _resolve_pending_action(FakeAlice(), Req("no"), "approver")

    assert settled.metadata["type"] == "approval_rejected"
    assert (workspace / "pkg" / "sample.py").read_text(encoding="utf-8") == "original\n"
    assert pending_actions.get("approver") is None


def test_an_unrelated_reply_leaves_the_request_outstanding(workspace):
    _request_approval(
        {"tool": "run_command", "arguments": {"command": "pip install requests"}, "reason": "command_not_allowlisted"},
        user_id="approver",
    )
    assert _resolve_pending_action(FakeAlice(), Req("what's the weather"), "approver") is None
    assert pending_actions.get("approver") is not None


def test_nothing_pending_means_yes_is_just_conversation(workspace):
    pending_actions.clear("approver")
    assert _resolve_pending_action(FakeAlice(), Req("yes"), "approver") is None


def test_expired_requests_are_not_executed(workspace):
    pending_actions.record(
        user_id="approver",
        tool="write_workspace_file",
        arguments={"path": "pkg/sample.py", "content": "x", "overwrite": True},
        ttl_seconds=30,
    )
    stored = pending_actions.get("approver")
    stored.expires_at = 0.0
    pending_actions._write_all({"approver": stored.to_dict()})

    assert pending_actions.get("approver") is None
    assert (workspace / "pkg" / "sample.py").read_text(encoding="utf-8") == "original\n"


@pytest.mark.parametrize("text", ["yes", "Yes.", "go ahead", "do it", "sure", "OK", "proceed"])
def test_approval_phrases(text):
    assert _is_approval_phrase(text)


@pytest.mark.parametrize("text", ["no", "nope", "cancel", "never mind", "forget it"])
def test_rejection_phrases(text):
    assert _is_rejection_phrase(text)


@pytest.mark.parametrize("text", ["yes please do the other thing", "maybe", "why"])
def test_ambiguous_replies_are_not_treated_as_approval(text):
    assert not _is_approval_phrase(text)


def test_action_descriptions_name_the_real_target():
    assert _describe_action("run_command", {"command": "pytest -q"}) == "run `pytest -q`"
    assert _describe_action("edit_workspace_file", {"path": "a/b.py"}) == "edit a/b.py"
    assert _describe_action("write_workspace_file", {"path": "a/b.py"}) == "create a/b.py"
    assert _describe_action("write_workspace_file", {"path": "a/b.py", "overwrite": True}) == "replace a/b.py"


class ScriptedLLM:
    def __init__(self, responses):
        self._responses = list(responses)

    def chat_with_tools(self, messages, tools=None, **kwargs):
        return self._responses.pop(0) if self._responses else ChatResponse(content="done")


def _call(name, arguments):
    return ChatResponse(
        content="",
        tool_calls=[ToolCall(name=name, arguments=arguments)],
        raw={"message": {"tool_calls": [{"function": {"name": name, "arguments": arguments}}]}},
    )


def test_checkpoint_restores_a_modified_file(workspace):
    """Rollback must return real content, not just report that it happened."""
    target = workspace / "pkg" / "sample.py"
    llm = ScriptedLLM(
        [
            _call("edit_workspace_file", {"path": "pkg/sample.py", "find": "original", "replace": "changed"}),
            ChatResponse(content="done"),
        ]
    )
    loop = ReactLoop(llm, allow_write_tools=True, checkpoint_writes=True)
    result = loop.run("edit it")
    assert target.read_text(encoding="utf-8") == "changed\n"

    assert loop.rollback(result) is True
    assert target.read_text(encoding="utf-8") == "original\n"


def test_checkpoint_deletes_a_file_that_did_not_exist_before(workspace):
    llm = ScriptedLLM(
        [_call("write_workspace_file", {"path": "pkg/brand_new.py", "content": "x\n"}), ChatResponse(content="done")]
    )
    loop = ReactLoop(llm, allow_write_tools=True, checkpoint_writes=True)
    result = loop.run("create it")
    created = workspace / "pkg" / "brand_new.py"
    assert created.exists()

    assert loop.rollback(result) is True
    assert not created.exists()


def test_checkpoint_never_touches_git(workspace, monkeypatch):
    """An earlier version ran `git stash`, which swallowed unrelated uncommitted work."""
    import ai.integration.git_manager as git_manager

    def explode(*args, **kwargs):
        raise AssertionError("the checkpoint must not run git commands")

    monkeypatch.setattr(git_manager, "get_git_manager", explode)

    llm = ScriptedLLM(
        [_call("write_workspace_file", {"path": "pkg/x.py", "content": "1\n"}), ChatResponse(content="done")]
    )
    ReactLoop(llm, allow_write_tools=True, checkpoint_writes=True).run("write it")


def test_failed_write_after_a_successful_one_triggers_rollback(workspace, monkeypatch):
    llm = ScriptedLLM(
        [
            _call("write_workspace_file", {"path": "pkg/new.py", "content": "ok\n"}),
            _call("edit_workspace_file", {"path": "pkg/missing.py", "find": "a", "replace": "b"}),
        ]
    )
    loop = ReactLoop(llm, allow_write_tools=True, checkpoint_writes=False)

    rolled_back = {}

    def fake_rollback(result):
        rolled_back["called"] = True
        result.rolled_back = True
        return True

    monkeypatch.setattr(loop, "rollback", fake_rollback)
    result = loop.run("do two writes, second one fails")

    assert rolled_back.get("called") is True
    assert result.stopped_reason == "rolled_back_after_failed_write"


def test_executed_tools_are_written_to_the_journal(workspace, monkeypatch):
    recorded = []

    class FakeJournal:
        def record(self, entry):
            recorded.append(entry)

    monkeypatch.setattr("ai.core.execution_journal.get_execution_journal", lambda *a, **k: FakeJournal())

    llm = ScriptedLLM([_call("list_workspace_files", {}), ChatResponse(content="ok")])
    ReactLoop(llm, checkpoint_writes=False).run("list files")

    assert recorded
    assert recorded[0]["action"] == "list_workspace_files"
    assert recorded[0]["status"] == "success"


def test_write_tools_reach_the_catalog_used_by_the_runtime():
    """The runtime enables writes, so the schema it sends must contain them."""
    names = {s["function"]["name"] for s in tc.build_tool_schemas(max_risk=tc.RISK_OUTWARD)}
    assert {"write_workspace_file", "edit_workspace_file", "run_command"} <= names
