import time
from pathlib import Path

from ai.planning.planner import ReasoningPlanner, StepStatus
from ai.planning.task import PersistentTaskQueue, TaskStatus


def test_reasoning_planner_decomposes_and_verifies() -> None:
    planner = ReasoningPlanner()
    task = planner.create_task_representation(
        "Build me a release checklist and verify it",
        constraints=["must be concise"],
    )
    plan = planner.create_plan(task)

    assert len(plan.steps) >= 3

    dec = planner.decide_next_step(plan)
    assert dec is not None
    assert dec.step_id == 1

    ok = planner.apply_step_result(plan, "S1", {"summary": "objective clarified"})
    assert ok is True
    assert next(s for s in plan.steps if s.step_id == 1).status == StepStatus.COMPLETED

    trace = planner.debug_trace_view(plan)
    assert "Thought=" in trace
    assert "Decision=" in trace


def test_reasoning_planner_complexity_scales_steps_and_critical_path() -> None:
    planner = ReasoningPlanner()
    task = planner.create_task_representation(
        "Design, implement, test, and deploy a robust release workflow while handling rollback and failure scenarios",
        constraints=["must include verification", "must include fallback"],
    )
    plan = planner.create_plan(task)

    assert len(plan.steps) >= 4
    assert planner.estimate_critical_path(plan) >= 2


def test_persistent_task_queue_runs_and_persists(tmp_path: Path) -> None:
    store = tmp_path / "tasks.json"
    queue = PersistentTaskQueue(str(store))

    queue.register_handler("echo", lambda task: {"echo": task.payload.get("text")})
    task = queue.create_task("echo", {"text": "hello"}, priority=1)

    ran = queue.run_once()
    assert ran is True

    tasks = queue.list_tasks()
    current = next(t for t in tasks if t.task_id == task.task_id)
    assert current.status == TaskStatus.COMPLETED
    assert isinstance(current.result, dict)
    assert current.result.get("echo") == "hello"

    reloaded = PersistentTaskQueue(str(store))
    tasks2 = reloaded.list_tasks()
    loaded = next(t for t in tasks2 if t.task_id == task.task_id)
    assert loaded.status == TaskStatus.COMPLETED


def test_persistent_task_queue_dependency_order(tmp_path: Path) -> None:
    store = tmp_path / "tasks.json"
    queue = PersistentTaskQueue(str(store))

    queue.register_handler("record", lambda task: task.payload.get("name"))
    first = queue.create_task("record", {"name": "first"}, priority=5)
    second = queue.create_task(
        "record",
        {"name": "second"},
        priority=1,
        dependencies=[first.task_id],
    )

    queue.run_once()
    queue.run_once()

    all_tasks = {t.task_id: t for t in queue.list_tasks()}
    assert all_tasks[first.task_id].status == TaskStatus.COMPLETED
    assert all_tasks[second.task_id].status == TaskStatus.COMPLETED


def test_persistent_task_queue_rejects_dependency_cycles(tmp_path: Path) -> None:
    store = tmp_path / "tasks.json"
    queue = PersistentTaskQueue(str(store))

    t1 = queue.create_task("echo", {"x": 1})
    # Create a dependent task first.
    t2 = queue.create_task("echo", {"x": 2}, dependencies=[t1.task_id])

    # Force a back-edge and verify cycle detector catches it.
    queue._tasks[t1.task_id].dependencies = [t2.task_id]
    assert queue._has_dependency_cycle() is True


def test_persistent_task_queue_background_loop(tmp_path: Path) -> None:
    store = tmp_path / "tasks.json"
    queue = PersistentTaskQueue(str(store))

    queue.register_handler("work", lambda task: {"done": True, "id": task.task_id})
    queue.create_task("work", {"x": 1}, priority=2)

    queue.start_background_loop(tick_seconds=0.05)
    time.sleep(0.2)
    queue.stop_background_loop()

    snap = queue.snapshot()
    assert snap["counts"][TaskStatus.COMPLETED.value] >= 1


def test_persistent_task_queue_serializes_entity_like_payload(tmp_path: Path) -> None:
    store = tmp_path / "tasks.json"
    queue = PersistentTaskQueue(str(store))

    class _EntityLike:
        def __init__(self, value: str):
            self.value = value

    queue.register_handler("echo", lambda task: {"payload": task.payload})
    created = queue.create_task(
        "echo",
        {"CATEGORY": [_EntityLike("work")], "query": "i want to work on an ai project"},
    )

    ran = queue.run_once()
    assert ran is True

    reloaded = PersistentTaskQueue(str(store))
    tasks = {t.task_id: t for t in reloaded.list_tasks()}
    loaded = tasks[created.task_id]
    assert loaded.status == TaskStatus.COMPLETED
    assert loaded.payload.get("CATEGORY") == ["work"]
