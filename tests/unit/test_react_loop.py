"""The reason/act/observe loop: what the model is offered, what actually runs.

These drive the loop against a scripted model (tests/support/fake_llm.py) rather
than a live Ollama, so the parts of Alice that decide what to *do* are covered
without a model being installed.
"""

import pytest

from ai.core import tool_catalog as catalog
from ai.core.react_loop import ReactLoop
from tests.support.fake_llm import FakeLLM, Turn, UnreachableLLM, answer, calls


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Point the tool catalog at a scratch project so file tools are harmless."""
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "plan.md").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    monkeypatch.setattr(catalog, "project_root", lambda: tmp_path)
    return tmp_path


def _tool_names(offered):
    return set(offered)


# -- what the model is allowed to see ---------------------------------------


def test_read_only_loop_is_not_offered_write_tools(workspace):
    """Advertising a tool the loop will then refuse loses the user's request:
    the model asks for the write, the turn is discarded as approval_required,
    and the user gets ordinary chat instead of either the edit or a prompt."""
    llm = FakeLLM([answer("nothing to do")])
    ReactLoop(llm, allow_write_tools=False).run("tidy up the workspace")

    offered = _tool_names(llm.tools_offered[0])
    assert "read_workspace_file" in offered
    assert "write_workspace_file" not in offered
    assert "edit_workspace_file" not in offered
    assert "run_command" not in offered


def test_write_enabled_loop_is_offered_write_tools(workspace):
    llm = FakeLLM([answer("nothing to do")])
    ReactLoop(llm, allow_write_tools=True).run("tidy up the workspace")

    offered = _tool_names(llm.tools_offered[0])
    assert "write_workspace_file" in offered
    assert "read_workspace_file" in offered


# -- multi-step grounding ----------------------------------------------------


def test_the_loop_chains_tools_and_feeds_results_back(workspace):
    """List, then read what the listing found, then answer from the contents."""
    llm = FakeLLM(
        [
            calls(("list_workspace_files", {"subdirectory": "notes"})),
            calls(("read_workspace_file", {"path": "notes/plan.md"})),
            answer("The plan lists alpha, beta and gamma."),
        ]
    )
    result = ReactLoop(llm).run("what is in my plan?")

    assert result.used_tools is True
    assert [step.tool for step in result.steps] == ["list_workspace_files", "read_workspace_file"]
    assert result.answer == "The plan lists alpha, beta and gamma."
    assert result.stopped_reason == "answered"

    # The second round trip must actually show the model what the first returned.
    assert any("plan.md" in obs for obs in llm.observations_at(1))
    # The third must carry the file contents, or the answer was not grounded.
    assert any("gamma" in obs for obs in llm.observations_at(2))


def test_an_answer_with_no_tool_call_reports_no_tool_use(workspace):
    llm = FakeLLM([answer("Morning.")])
    result = ReactLoop(llm).run("morning")

    assert result.used_tools is False
    assert result.steps == []
    assert result.answer == "Morning."


# -- repeats -----------------------------------------------------------------


def test_a_repeated_call_is_not_executed_twice(workspace, monkeypatch):
    """The repeat guard used to run after execution, so a command asked for
    twice ran twice and the model was then told to ignore the second result."""
    executed = []
    real_execute = catalog.execute_tool

    def counting_execute(name, arguments=None, **kwargs):
        executed.append(name)
        return real_execute(name, arguments, **kwargs)

    monkeypatch.setattr(catalog, "execute_tool", counting_execute)

    same = ("read_workspace_file", {"path": "notes/plan.md"})
    llm = FakeLLM([calls(same), calls(same), answer("alpha, beta, gamma")])
    result = ReactLoop(llm).run("read the plan twice")

    assert executed == ["read_workspace_file"]
    assert result.answer == "alpha, beta, gamma"


def test_differing_arguments_are_not_treated_as_a_repeat(workspace):
    (workspace / "notes" / "other.md").write_text("delta\n", encoding="utf-8")
    llm = FakeLLM(
        [
            calls(("read_workspace_file", {"path": "notes/plan.md"})),
            calls(("read_workspace_file", {"path": "notes/other.md"})),
            answer("both read"),
        ]
    )
    result = ReactLoop(llm).run("read both notes")
    assert [s.tool for s in result.steps] == ["read_workspace_file", "read_workspace_file"]


# -- budgets and failure -----------------------------------------------------


def test_the_step_budget_bounds_tool_use(workspace):
    llm = FakeLLM(
        on_call=lambda index, messages: calls(("read_workspace_file", {"path": f"notes/{index}.md"})),
    )
    result = ReactLoop(llm, max_steps=3).run("keep going")
    assert len(result.steps) == 3
    assert result.stopped_reason == "step_budget_exhausted"


def test_a_dead_model_stops_the_loop_without_raising(workspace):
    result = ReactLoop(UnreachableLLM()).run("what is in my plan?")
    assert result.stopped_reason == "model_error"
    assert result.used_tools is False
    assert result.answer == ""


def test_a_failed_closing_call_is_reported_not_silently_empty(workspace):
    """An empty answer reads to the caller as 'the model declined', which is a
    different thing from 'the model was unreachable'."""

    def script(index, messages):
        if index == 0:
            return calls(("read_workspace_file", {"path": "notes/plan.md"}))
        raise ConnectionError("model went away")

    result = ReactLoop(FakeLLM(on_call=script)).run("what is in my plan?")
    assert result.used_tools is True
    assert result.answer == ""
    assert result.final_answer_failed is True


def test_a_failed_closing_call_does_not_mask_why_the_loop_stopped():
    """step_budget_exhausted says more about the turn than a generic closing
    failure, so it has to survive."""
    llm = FakeLLM(on_call=lambda index, messages: calls(("get_system_status", {})))
    result = ReactLoop(llm, max_steps=2).run("keep going")
    assert result.stopped_reason == "step_budget_exhausted"


def test_an_unknown_tool_is_reported_back_to_the_model(workspace):
    llm = FakeLLM([calls(("teleport", {})), answer("I cannot do that.")])
    result = ReactLoop(llm).run("teleport me")

    assert result.steps[0].error == "unknown tool"
    assert any("unknown tool" in obs for obs in llm.observations_at(1))


# -- the trust boundary ------------------------------------------------------


def test_a_refused_tool_stops_the_loop(workspace):
    llm = FakeLLM([calls(("run_command", {"command": "rm -rf /"}))])
    result = ReactLoop(llm, allow_write_tools=True).run("clean up")

    assert result.stopped_reason == "refused"
    assert result.refused["tool"] == "run_command"
    assert result.used_tools is False


def test_a_write_outside_the_workspace_asks_before_running(workspace):
    llm = FakeLLM([calls(("write_workspace_file", {"path": "/etc/passwd", "content": "x"}))])
    result = ReactLoop(llm, allow_write_tools=True, checkpoint_writes=False).run("write that file")

    assert result.stopped_reason == "approval_required"
    assert result.pending_approval["tool"] == "write_workspace_file"
    assert result.pending_approval["reason"] == "writes_outside_workspace"


def test_a_workspace_write_runs_and_lands_on_disk(workspace):
    llm = FakeLLM(
        [
            calls(("write_workspace_file", {"path": "notes/new.md", "content": "hello"})),
            answer("Written."),
        ]
    )
    result = ReactLoop(llm, allow_write_tools=True, checkpoint_writes=False).run("save a note saying hello")

    assert result.steps[0].success is True
    assert (workspace / "notes" / "new.md").read_text(encoding="utf-8") == "hello"
    assert result.answer == "Written."


def test_context_is_passed_to_the_model_as_a_system_message(workspace):
    llm = FakeLLM([answer("ok")])
    ReactLoop(llm).run("what did we decide?", context="Earlier: we chose SQLite.")

    system_messages = [m["content"] for m in llm.calls[0]["messages"] if m["role"] == "system"]
    assert any("we chose SQLite" in m for m in system_messages)


def test_narrowing_the_tool_set_is_respected(workspace):
    llm = FakeLLM([answer("ok")])
    ReactLoop(llm).run("anything", tool_names=["read_workspace_file"])
    assert llm.tools_offered[0] == ["read_workspace_file"]


def test_no_tools_available_is_reported(workspace):
    llm = FakeLLM([answer("ok")])
    result = ReactLoop(llm).run("anything", tool_names=["not_a_real_tool"])
    assert result.stopped_reason == "no_tools_available"
    assert llm.calls == []


def test_unproductive_lookups_are_marked_so_they_cannot_hijack_the_turn(workspace):
    """A passing remark that sends the model to a tool which finds nothing must
    not turn the whole reply into 'there are no results'."""
    llm = FakeLLM(
        [
            calls(("search_workspace", {"query": "zzzzz-not-present"})),
            answer("Nothing matched."),
        ]
    )
    result = ReactLoop(llm).run("anything about zzzzz?")
    assert result.steps[0].productive is False
    assert result.produced_evidence is False


def test_a_productive_lookup_counts_as_evidence(workspace):
    llm = FakeLLM(
        [
            calls(("read_workspace_file", {"path": "notes/plan.md"})),
            answer("alpha, beta, gamma"),
        ]
    )
    result = ReactLoop(llm).run("read my plan")
    assert result.steps[0].productive is True
    assert result.produced_evidence is True


def test_turn_helper_builds_the_shape_ollama_returns():
    turn = calls(("get_system_status", {}))
    assert isinstance(turn, Turn)
    assert turn.tool_calls[0].name == "get_system_status"
