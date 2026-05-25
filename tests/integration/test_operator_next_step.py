from __future__ import annotations

from ai.runtime.next_step_policy import decide_next_step
from ai.runtime.response_momentum_policy import apply_response_momentum
from ai.core.routing.evidence_contracts import EvidenceContracts


def test_next_step_from_durable_state_is_grounded():
    decision = decide_next_step(
        route="local",
        intent="operator:next_step",
        operator_state={
            "active_objective": "Improve agentic companion operator runtime",
            "last_failure": "routing misroute in next-step handling",
            "files_inspected": ["ai/runtime/contract_pipeline.py"],
        },
        project_memory={
            "active_objective": "Improve agentic companion operator runtime"
        },
        local_execution={},
        available_files=[
            "ai/core/routing/route_arbiter.py",
            "ai/core/routing/evidence_contracts.py",
            "ai/runtime/routing_adapter.py",
        ],
        files_inspected=["ai/runtime/contract_pipeline.py"],
        last_failure="routing misroute in next-step handling",
    )
    payload = decision.to_dict()
    # next_recommended_action is raw action text — no "Next best move:" label now
    assert payload["next_recommended_action"]
    assert "route_arbiter.py" in payload["next_recommended_action"].lower() or any(
        "route_arbiter.py" in f for f in payload["suggested_next_files"]
    )
    assert payload["candidate_actions"]
    assert any("route_arbiter.py" in f for f in payload["suggested_next_files"])


def test_response_momentum_includes_next_move_without_passive_closer():
    text = apply_response_momentum(
        response_text="Inspected routing contracts.",
        intent="code:analyze_file",
        route="local",
        operator_state={
            "active_objective": "Improve agentic companion operator runtime",
            "current_focus": "routing",
        },
        project_memory={},
        local_execution={
            "action": "code:analyze_file",
            "inspected_file": "ai/core/routing/route_arbiter.py",
        },
        next_step="Inspect ai/core/routing/evidence_contracts.py for missing slot scoring.",
    )
    low = text.lower()
    # Natural next-step pointer should appear without the "Next best move:" label
    assert "next best move" not in low
    assert "evidence_contracts.py" in low
    assert "let me know if you need anything else" not in low


def test_evidence_contract_blocks_file_read_without_target():
    result = EvidenceContracts.evaluate(
        intent="file_operations:read",
        user_input="read a file",
    )
    assert result["accepted"] is False
    assert "file_target" in list(result.get("missing_slots") or [])
