from types import SimpleNamespace

from ai.core.agentic_loop import AgentCycleInput, AgenticLoop


class _ActionEngineStub:
    def __init__(self, success: bool = True):
        self.success = success

    def execute(self, request):
        return SimpleNamespace(
            status="success" if self.success else "failed",
            success=self.success,
            goal_satisfied=self.success,
            recovery_path="retry" if not self.success else "done",
            state_updates={"request_plugin": request.plugin},
        )


def test_agentic_loop_executes_when_goal_and_intent_are_present():
    loop = AgenticLoop(action_engine=_ActionEngineStub(success=True))
    result = loop.run_cycle(
        AgentCycleInput(
            user_input="check weather in Austin",
            intent="weather:current",
            confidence=0.88,
            entities={"target": "Austin"},
            long_horizon_goal="Keep weather awareness current",
        )
    )
    assert result.execution["success"] is True
    assert result.decision["plugin"] == "weather"
    assert result.orchestration["next_milestone"] == "execute_primary_action"


def test_agentic_loop_skips_execution_without_goal():
    loop = AgenticLoop(action_engine=_ActionEngineStub(success=True))
    result = loop.run_cycle(
        AgentCycleInput(
            user_input="",
            intent="weather:current",
            confidence=0.4,
            entities={},
            long_horizon_goal="",
        )
    )
    assert result.execution["status"] == "skipped"
    assert result.decision["should_execute"] is False
