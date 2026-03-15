"""
Integration tests for reasoning -> planning -> execution layer.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai.planning.task_planner import TaskPlanner
from ai.planning.plan_executor import PlanExecutor


class _DummyPluginManager:
    def get_plugin(self, _name):
        return None


class _DummyLLM:
    def generate(self, prompt, max_tokens=None):
        return f"LLM:{prompt[:60]}"


class _DummyMemory:
    def search(self, query, top_k=3):
        return [{"query": query, "top_k": top_k}]


def test_study_topic_plan_has_reasoning_steps() -> None:
    planner = TaskPlanner()

    plan = planner.create_plan(
        intent="learning:study_topic",
        entities={"topic": "polymorphism", "query": "help me study polymorphism"},
        context={},
    )

    actions = [step.action for step in plan.steps]
    assert actions == [
        "response.explain",
        "response.example",
        "response.check_understanding",
        "response.deeper_material",
    ]
    assert planner.validate_plan(plan) is True


def test_study_topic_plan_executes_all_steps() -> None:
    planner = TaskPlanner()
    executor = PlanExecutor(
        plugin_manager=_DummyPluginManager(),
        llm_engine=_DummyLLM(),
        memory_system=_DummyMemory(),
    )

    plan = planner.create_plan(
        intent="learning:study_topic",
        entities={"topic": "polymorphism", "query": "help me study polymorphism"},
        context={},
    )

    result = executor.execute(plan)

    assert result["success"] is True
    assert set(result["all_results"].keys()) == {1, 2, 3, 4}
    assert "polymorphism" in result["all_results"][2].lower()
    assert "quick check" in result["all_results"][3].lower()
