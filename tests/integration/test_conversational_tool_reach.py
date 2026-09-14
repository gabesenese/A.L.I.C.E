"""Whether the model is allowed to reach for a tool on an ordinary turn.

The tool catalog has always carried weather, notes and system-state tools, but
only codebase turns ever reached the agent loop. Everything else could act only
when the keyword router matched a known intent and dispatched a plugin, so a
question phrased outside those patterns got a chat reply about the thing instead
of the thing itself.
"""

from types import SimpleNamespace

import pytest

from ai.core import tool_catalog as catalog
from ai.runtime.boundaries import boundary_factory
from tests.support.fake_llm import FakeLLM, UnreachableLLM, answer, calls


def _req(route="llm", intent="conversation:general", user_input="what is in my plan?"):
    return SimpleNamespace(
        user_input=user_input,
        decision=SimpleNamespace(route=route, intent=intent, metadata={}),
    )


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "plan.md").write_text("ship the memory fix\n", encoding="utf-8")
    monkeypatch.setattr(catalog, "project_root", lambda: tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def _enable_flag(monkeypatch):
    monkeypatch.delenv("ALICE_ENABLE_CONVERSATIONAL_TOOL_USE", raising=False)


# -- the gate ---------------------------------------------------------------


def test_a_codebase_turn_may_reach_for_tools():
    assert boundary_factory._may_reach_for_tools(_req(route="local", intent="code:request")) is True


def test_an_ordinary_conversational_turn_may_now_reach_for_tools():
    assert boundary_factory._may_reach_for_tools(_req()) is True


@pytest.mark.parametrize("route", ["tool", "plugin"])
def test_a_turn_already_dispatched_to_a_plugin_is_left_alone(route):
    """That dispatch has the answer in hand; a second opinion is a wasted round trip."""
    assert boundary_factory._may_reach_for_tools(_req(route=route, intent="weather:current")) is False


def test_conversational_reach_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("ALICE_ENABLE_CONVERSATIONAL_TOOL_USE", "0")
    assert boundary_factory._may_reach_for_tools(_req()) is False
    # A codebase turn is not governed by the flag.
    assert boundary_factory._may_reach_for_tools(_req(route="local", intent="code:request")) is True


# -- the grounded answer ----------------------------------------------------


def test_a_conversational_turn_can_now_be_answered_from_a_real_lookup(workspace):
    llm = FakeLLM(
        [
            calls(("read_workspace_file", {"path": "notes/plan.md"})),
            answer("Your plan says to ship the memory fix."),
        ]
    )
    alice = SimpleNamespace(llm=llm, plugins=None)

    out = boundary_factory._try_tool_grounded_answer(alice, _req(), {})

    assert out is not None
    assert out.text == "Your plan says to ship the memory fix."
    assert out.metadata["type"] == "tool_grounded_answer"
    assert out.metadata["tools_used"] == ["read_workspace_file"]


def test_a_turn_the_model_answers_without_tools_falls_through_to_conversation(workspace):
    """Chat has to stay chat: the loop declining to act must not become the reply."""
    llm = FakeLLM([answer("I think so, yes.")])
    alice = SimpleNamespace(llm=llm, plugins=None)

    assert boundary_factory._try_tool_grounded_answer(alice, _req(), {}) is None


def test_a_lookup_that_finds_nothing_falls_through_to_conversation(workspace):
    """Otherwise a passing remark sends the model to a tool, the tool finds
    nothing, and 'there are no results' becomes the whole turn."""
    llm = FakeLLM(
        [
            calls(("search_workspace", {"query": "zzzz-absent"})),
            answer("There is nothing about that."),
        ]
    )
    alice = SimpleNamespace(llm=llm, plugins=None)

    assert boundary_factory._try_tool_grounded_answer(alice, _req(), {}) is None


def test_a_dead_model_falls_through_instead_of_raising(workspace):
    alice = SimpleNamespace(llm=UnreachableLLM(), plugins=None)
    assert boundary_factory._try_tool_grounded_answer(alice, _req(), {}) is None


def test_an_llm_without_tool_support_falls_through(workspace):
    alice = SimpleNamespace(llm=SimpleNamespace(chat=lambda *a, **k: "hi"), plugins=None)
    assert boundary_factory._try_tool_grounded_answer(alice, _req(), {}) is None


# -- budgets ----------------------------------------------------------------


def test_a_conversational_turn_gets_a_tighter_step_budget_than_a_codebase_turn(workspace):
    """A mid-conversation lookup is one or two calls. The user is waiting either
    way, so a conversational turn must not spiral the way a code task may."""
    seen = {}

    def record(index, messages):
        seen.setdefault("calls", 0)
        seen["calls"] += 1
        return calls(("read_workspace_file", {"path": f"notes/{index}.md"}))

    alice = SimpleNamespace(llm=FakeLLM(on_call=record), plugins=None)
    boundary_factory._try_tool_grounded_answer(alice, _req(), {})
    conversational_calls = seen["calls"]

    seen.clear()
    alice = SimpleNamespace(llm=FakeLLM(on_call=record), plugins=None)
    boundary_factory._try_tool_grounded_answer(alice, _req(route="local", intent="code:request"), {})
    workspace_calls = seen["calls"]

    assert conversational_calls < workspace_calls


def test_read_only_tools_are_what_a_conversational_turn_is_offered(workspace):
    llm = FakeLLM([answer("ok")])
    alice = SimpleNamespace(llm=llm, plugins=None)
    boundary_factory._try_tool_grounded_answer(alice, _req(), {})

    offered = set(llm.tools_offered[0])
    assert "get_current_weather" in offered
    assert "list_notes" in offered
    assert "get_system_status" in offered
    # Nothing that writes or leaves the machine runs without a confirmation.
    assert "write_workspace_file" not in offered
    assert "run_command" not in offered
