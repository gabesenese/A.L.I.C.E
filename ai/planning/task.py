"""
Persistent task system with queue, priorities, and background execution loop.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    task_id: str
    kind: str
    payload: Dict[str, Any]
    priority: int = 5
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    result: Any = None
    error: str = ""
    attempts: int = 0
    max_attempts: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "kind": self.kind,
            "payload": self.payload,
            "priority": int(self.priority),
            "dependencies": list(self.dependencies),
            "status": self.status.value,
            "created_at": float(self.created_at),
            "updated_at": float(self.updated_at),
            "result": self.result,
            "error": self.error,
            "attempts": int(self.attempts),
            "max_attempts": int(self.max_attempts),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        status_raw = str(data.get("status") or TaskStatus.PENDING.value)
        try:
            status = TaskStatus(status_raw)
        except ValueError:
            status = TaskStatus.PENDING
        return cls(
            task_id=str(data.get("task_id") or f"task-{uuid.uuid4().hex[:8]}"),
            kind=str(data.get("kind") or "generic"),
            payload=dict(data.get("payload") or {}),
            priority=int(data.get("priority") or 5),
            dependencies=[str(d) for d in (data.get("dependencies") or [])],
            status=status,
            created_at=float(data.get("created_at") or time.time()),
            updated_at=float(data.get("updated_at") or time.time()),
            result=data.get("result"),
            error=str(data.get("error") or ""),
            attempts=int(data.get("attempts") or 0),
            max_attempts=max(1, int(data.get("max_attempts") or 1)),
        )


class PersistentTaskQueue:
    """Priority queue with JSON persistence and optional background worker."""

    def __init__(self, persistence_path: str = "data/planning/tasks.json") -> None:
        self.persistence_path = Path(persistence_path)
        self._tasks: Dict[str, Task] = {}
        self._handlers: Dict[str, Callable[[Task], Any]] = {}
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._tick_seconds = 0.25
        self._load()

    def register_handler(self, kind: str, handler: Callable[[Task], Any]) -> None:
        self._handlers[kind] = handler

    def create_task(
        self,
        kind: str,
        payload: Dict[str, Any],
        priority: int = 5,
        *,
        dependencies: Optional[List[str]] = None,
        max_attempts: int = 1,
    ) -> Task:
        deps = [str(d) for d in (dependencies or [])]
        task = Task(
            task_id=f"task-{uuid.uuid4().hex[:10]}",
            kind=kind,
            payload=dict(payload or {}),
            priority=max(0, min(9, int(priority))),
            dependencies=deps,
            max_attempts=max(1, int(max_attempts)),
        )
        with self._lock:
            missing = [dep for dep in deps if dep not in self._tasks]
            if missing:
                raise ValueError(f"Unknown dependencies: {missing}")
            self._tasks[task.task_id] = task
            if self._has_dependency_cycle():
                del self._tasks[task.task_id]
                raise ValueError("Task dependency cycle detected")
            self._save()
        return task

    def list_tasks(self, status: Optional[TaskStatus] = None) -> List[Task]:
        with self._lock:
            tasks = list(self._tasks.values())
        if status is not None:
            tasks = [t for t in tasks if t.status == status]
        return sorted(tasks, key=lambda t: (t.priority, t.created_at))

    def next_pending_task(self) -> Optional[Task]:
        with self._lock:
            completed = {
                t.task_id
                for t in self._tasks.values()
                if t.status == TaskStatus.COMPLETED
            }
            candidates = [
                t
                for t in self._tasks.values()
                if t.status == TaskStatus.PENDING and self._deps_satisfied(t, completed)
            ]
        if not candidates:
            return None
        now = time.time()
        return min(
            candidates,
            key=lambda t: (self._effective_priority(t, now), t.created_at, t.task_id),
        )

    def mark_task(
        self, task_id: str, status: TaskStatus, result: Any = None, error: str = ""
    ) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            task.status = status
            task.updated_at = time.time()
            task.result = result
            task.error = error
            self._save()
        return True

    def run_once(self) -> bool:
        task = self.next_pending_task()
        if task is None:
            return False

        self.mark_task(task.task_id, TaskStatus.RUNNING)
        handler = self._handlers.get(task.kind)
        if handler is None:
            self.mark_task(
                task.task_id,
                TaskStatus.FAILED,
                error=f"No handler registered for task kind '{task.kind}'",
            )
            return True

        try:
            result = handler(task)
            self.mark_task(task.task_id, TaskStatus.COMPLETED, result=result)
        except Exception as exc:
            with self._lock:
                current = self._tasks.get(task.task_id)
                if current is None:
                    return True
                current.attempts += 1
                current.updated_at = time.time()
                if current.attempts < current.max_attempts:
                    current.status = TaskStatus.PENDING
                    current.error = str(exc)
                else:
                    current.status = TaskStatus.FAILED
                    current.error = str(exc)
                self._save()
        return True

    def start_background_loop(self, tick_seconds: float = 0.25) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._tick_seconds = max(0.05, float(tick_seconds or 0.25))

        self._thread = threading.Thread(
            target=self._loop, name="persistent-task-queue", daemon=True
        )
        self._thread.start()

    def stop_background_loop(self, timeout: float = 2.0) -> None:
        with self._lock:
            self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def snapshot(self) -> Dict[str, Any]:
        tasks = self.list_tasks()
        counts = {
            TaskStatus.PENDING.value: 0,
            TaskStatus.RUNNING.value: 0,
            TaskStatus.COMPLETED.value: 0,
            TaskStatus.FAILED.value: 0,
        }
        for task in tasks:
            counts[task.status.value] = counts.get(task.status.value, 0) + 1
        return {
            "total": len(tasks),
            "counts": counts,
            "ready_pending": int(
                sum(
                    1
                    for t in tasks
                    if t.status == TaskStatus.PENDING
                    and self._deps_satisfied(
                        t,
                        {x.task_id for x in tasks if x.status == TaskStatus.COMPLETED},
                    )
                )
            ),
            "tasks": [t.to_dict() for t in tasks],
        }

    def _effective_priority(self, task: Task, now: float) -> float:
        # Aging prevents starvation: older pending tasks gradually move up.
        wait_seconds = max(0.0, now - float(task.created_at))
        aging_bonus = min(2.0, wait_seconds / 300.0)
        return float(task.priority) - aging_bonus

    def _deps_satisfied(self, task: Task, completed: set[str]) -> bool:
        return all(dep in completed for dep in (task.dependencies or []))

    def _has_dependency_cycle(self) -> bool:
        graph = {
            task_id: list(task.dependencies or [])
            for task_id, task in self._tasks.items()
        }
        seen: set[str] = set()
        in_stack: set[str] = set()

        def dfs(node: str) -> bool:
            if node in in_stack:
                return True
            if node in seen:
                return False
            seen.add(node)
            in_stack.add(node)
            for dep in graph.get(node, []):
                if dep in graph and dfs(dep):
                    return True
            in_stack.remove(node)
            return False

        return any(dfs(node) for node in graph)

    def _loop(self) -> None:
        while True:
            with self._lock:
                if not self._running:
                    break
            self.run_once()
            time.sleep(self._tick_seconds)

    def _load(self) -> None:
        with self._lock:
            if not self.persistence_path.exists():
                return
            try:
                raw = json.loads(self.persistence_path.read_text(encoding="utf-8"))
            except Exception:
                return
            if not isinstance(raw, list):
                return
            self._tasks = {}
            for item in raw:
                if isinstance(item, dict):
                    task = Task.from_dict(item)
                    self._tasks[task.task_id] = task

    def _save(self) -> None:
        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [t.to_dict() for t in self._tasks.values()]
        self.persistence_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
