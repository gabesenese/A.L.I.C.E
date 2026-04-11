from datetime import datetime, timedelta

from ai.core.executive_controller import ExecutiveController
from ai.core.execution_verifier import get_execution_verifier
from ai.core.goal_recognizer import get_goal_recognizer
from ai.core.live_state_service import get_live_state_service


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


def test_executive_state_machine_is_single_route_authority_for_tool_turns():
    controller = ExecutiveController()
    state = controller.build_state(
        user_input="delete note groceries",
        intent="notes:delete",
        confidence=0.90,
        entities={},
        conversation_state={},
    )
    decision = controller.decide(
        state,
        is_pure_conversation=False,
        has_explicit_action_cue=True,
        has_active_goal=False,
        force_plugins_for_notes=False,
    )

    state_machine = controller.run_turn_state_machine(
        state=state,
        decision=decision,
        has_explicit_action_cue=True,
        has_active_goal=False,
        pre_route_blocked=False,
        tool_vetoed=False,
    )

    assert state_machine.chosen_route == "tool"
    assert state_machine.should_try_plugins is True
    assert state_machine.contract.chosen_route == "tool"


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
