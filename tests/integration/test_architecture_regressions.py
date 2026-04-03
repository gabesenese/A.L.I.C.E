from datetime import datetime, timedelta

from ai.core.execution_verifier import get_execution_verifier
from ai.core.goal_recognizer import get_goal_recognizer
from ai.core.live_state_service import get_live_state_service
from ai.core.turn_routing_policy import get_turn_routing_policy


class _Entity:
    def __init__(self, data, created_at):
        self.data = data
        self.created_at = created_at


class _ReasoningEngine:
    def __init__(self, current=None, forecast=None):
        self._map = {
            "current_weather": current,
            "weather_forecast": forecast,
        }

    def get_entity(self, entity_id):
        return self._map.get(entity_id)


class _WorldStateMemory:
    def __init__(self, payload):
        self._payload = payload

    def get_environment_state(self, key):
        if key != "weather":
            return {}
        return dict(self._payload)


def test_goal_recognizer_detects_objective_statement():
    recognizer = get_goal_recognizer()
    signal = recognizer.detect(
        "My objective is to make this a reliable autonomous agent with verification loops"
    )

    assert signal is not None
    assert "autonomous" in " ".join(signal.markers)
    assert signal.project_direction == "agentic_autonomy"


def test_goal_recognizer_rejects_immediate_tool_command():
    recognizer = get_goal_recognizer()
    signal = recognizer.detect("I want to send an email and delete old notes")

    assert signal is None


def test_turn_routing_policy_has_single_owner_decision():
    policy = get_turn_routing_policy()
    decision = policy.decide(
        executive_action="use_plugin",
        runtime_allow_tools=True,
        runtime_preference="tool_first",
        is_short_followup=True,
        is_pure_conversation=False,
        force_plugins_for_notes=False,
    )

    assert decision.owner == "executive_turn_routing_policy"
    assert decision.should_try_plugins is True


def test_live_state_service_prefers_freshest_world_snapshot():
    service = get_live_state_service()
    now = datetime.now()
    stale_entity = _Entity(
        data={
            "location": "Kitchener",
            "condition": "rain",
            "temperature": 9,
            "captured_at": (now - timedelta(hours=2)).isoformat(),
        },
        created_at=now - timedelta(hours=2),
    )
    reasoning = _ReasoningEngine(current=stale_entity, forecast=None)
    world = _WorldStateMemory(
        {
            "data": {
                "location": "Kitchener",
                "condition": "clear sky",
                "temperature": 14,
                "captured_at": now.isoformat(),
            },
            "captured_at": now.isoformat(),
        }
    )

    snapshot = service.freshest_weather_snapshot(
        reasoning_engine=reasoning,
        world_state_memory=world,
    )

    assert snapshot is not None
    assert snapshot.get("source") == "world_state_memory:weather"
    assert snapshot.get("data", {}).get("condition") == "clear sky"


def test_execution_verifier_rejects_empty_planned_result():
    verifier = get_execution_verifier()
    report = verifier.verify_task_result(
        intent="learning:study_topic",
        result="",
        all_results={1: "", 2: "", 3: ""},
    )

    assert report.accepted is False
    assert "empty_result" in report.issues
