"""Asking for the HTN planner first must not deadlock on the habit miner."""

import threading

import ai.learning.pattern_miner as pm


def test_htn_planner_can_be_created_before_the_habit_miner(monkeypatch, tmp_path):
    monkeypatch.setattr(pm, "_habit_miner_instance", None)
    monkeypatch.setattr(pm, "_htn_planner_instance", None)
    monkeypatch.chdir(tmp_path)

    result = {}
    worker = threading.Thread(target=lambda: result.setdefault("planner", pm.get_htn_planner()), daemon=True)
    worker.start()
    worker.join(timeout=5)

    assert not worker.is_alive(), "get_htn_planner deadlocked waiting for its own lock"
    assert result["planner"] is pm.get_htn_planner()
