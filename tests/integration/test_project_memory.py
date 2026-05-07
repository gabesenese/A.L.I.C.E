from __future__ import annotations

from ai.memory.project_memory import (
    load_project_state,
    record_failure,
    record_file_inspected,
    save_project_state,
    update_project_state,
    ProjectMemoryState,
)
from ai.runtime.lifecycle_manager import LifecycleManager


def test_project_memory_stores_objective_and_focus():
    user_id = "test_project_memory_obj"
    save_project_state(ProjectMemoryState(), user_id=user_id)
    update_project_state(
        {
            "active_objective": "Improve agentic companion operator runtime",
            "current_focus": "runtime",
        },
        user_id=user_id,
    )
    state = load_project_state(user_id)
    assert "agentic companion operator runtime" in state.active_objective.lower()
    assert state.current_focus == "runtime"
    assert any("no voice for now" in c.lower() for c in state.design_constraints)


def test_project_memory_tracks_blocker_and_last_inspected_file():
    user_id = "test_project_memory_blocker"
    save_project_state(ProjectMemoryState(), user_id=user_id)
    record_failure("routing", "misroute during operator next-step", user_id=user_id)
    record_file_inspected("ai/core/routing/route_arbiter.py", user_id=user_id)
    state = load_project_state(user_id)
    assert "misroute" in state.last_failure.lower()
    assert state.files_inspected[-1] == "ai/core/routing/route_arbiter.py"


def test_lifecycle_manager_shutdown_skips_unset_systems():
    manager = LifecycleManager()
    report = manager.stop_optional_systems(
        {
            "continuous_learning": None,
            "proactive_loops": None,
            "voice": None,
        }
    )
    payload = report.to_dict()
    assert "continuous_learning" in payload["skipped"]
    assert payload["errors"] == []
