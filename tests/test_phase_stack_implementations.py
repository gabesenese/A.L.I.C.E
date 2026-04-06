from ai.core.foundation_layers import FoundationLayers
from ai.core.live_state_service import LiveStateService
from ai.core.unified_action_engine import ActionRequest, UnifiedActionEngine
from ai.runtime.user_state_model import UserStateModel
from tools.auditing.core_benchmark_gate import build_kpi_snapshot


class _WorldStateStub:
    def __init__(self, payload):
        self._payload = payload

    def get_environment_state(self, key: str):
        if key == "weather":
            return self._payload
        return {}


def test_foundation_layer_threshold_tuning():
    layers = FoundationLayers()
    before = layers.clarification_policy(plugin_scores={"notes": 1.0}, confidence=0.5)
    tuned = layers.tune_from_outcome(false_clarification=True)
    after = layers.clarification_policy(plugin_scores={"notes": 1.0}, confidence=0.5)

    assert tuned["confidence"] < before["thresholds"]["confidence"]
    assert after["thresholds"]["confidence"] == tuned["confidence"]


def test_live_state_reports_staleness_metadata():
    service = LiveStateService()
    stub = _WorldStateStub(
        {
            "captured_at": 1.0,
            "data": {"temp_c": 22},
        }
    )

    snapshot = service.freshest_weather_snapshot(world_state_memory=stub, max_age_seconds=10)
    assert snapshot is not None
    assert "is_stale" in snapshot
    assert snapshot["is_stale"] is True


def test_unified_action_engine_recovery_recommendation():
    engine = UnifiedActionEngine()
    request = ActionRequest(
        goal="delete note",
        plugin="notes",
        action="delete",
        params={},
        source_intent="notes:delete",
        confidence=0.6,
        retry_budget=0,
    )
    rec = engine._recommend_recovery_plan(
        request=request, result_dict={"success": False}, attempt_idx=0
    )
    assert rec["next_step"] == "request_target_clarification"


def test_user_state_preferences_update():
    model = UserStateModel()
    state = model.update_preferences(
        user_id="u1",
        response_brevity="concise",
        confirmation_style="minimal",
        risk_tolerance="low",
    )
    assert state.preferences["response_brevity"] == "concise"
    assert state.preferences["confirmation_style"] == "minimal"
    assert state.preferences["risk_tolerance"] == "low"


def test_benchmark_kpi_snapshot_builder():
    snapshot = build_kpi_snapshot(
        {
            "generated_at": "2026-04-06T00:00:00",
            "objective_score": 0.8,
            "intent_accuracy": 0.82,
            "useful_response_rate": 0.78,
            "latency_ms": {"p95": 130.0},
            "per_domain": {"notes": {"pass_rate": 0.9}},
        }
    )
    assert snapshot["objective_score"] == 0.8
    assert snapshot["latency_p95_ms"] == 130.0
