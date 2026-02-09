"""
Advanced Task Scheduler
========================
Sophisticated task scheduling with priority queues, dependency resolution,
parallel execution, and deadline management.
"""

import heapq
import time
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import threading
import logging

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"  # Waiting for dependencies


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


@dataclass
class Task:
    """Represents a scheduled task"""
    id: str
    name: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)

    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING

    dependencies: Set[str] = field(default_factory=set)
    deadline: Optional[datetime] = None
    estimate_duration: float = 0.0  # Estimated execution time in seconds

    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    completed: Optional[float] = None

    result: Any = None
    error: Optional[Exception] = None
    retry_count: int = 0
    max_retries: int = 3

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other):
        """Compare tasks for priority queue (lower priority value = higher priority)"""
        # First by priority level
        if self.priority != other.priority:
            return self.priority.value < other.priority.value

        # Then by deadline (earlier deadline first)
        if self.deadline and other.deadline:
            return self.deadline < other.deadline
        elif self.deadline:
            return True  # Has deadline beats no deadline

        # Finally by creation time (FIFO)
        return self.created < other.created

    def is_ready(self, completed_tasks: Set[str]) -> bool:
        """Check if all dependencies are satisfied"""
        return self.dependencies.issubset(completed_tasks)

    def is_overdue(self) -> bool:
        """Check if task is past deadline"""
        if self.deadline is None:
            return False
        return datetime.now() > self.deadline

    def duration(self) -> float:
        """Get actual execution duration"""
        if self.started and self.completed:
            return self.completed - self.started
        return 0.0


class TaskScheduler:
    """
    Advanced task scheduler with:
    - Priority-based scheduling
    - Dependency resolution
    - Deadline management
    - Parallel execution support
    - Task retry logic
    - Resource management
    """

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers

        # Task storage
        self.tasks: Dict[str, Task] = {}

        # Priority queue for ready tasks
        self.ready_queue: List[Task] = []

        # Task tracking
        self.completed_tasks: Set[str] = set()
        self.running_tasks: Set[str] = set()
        self.failed_tasks: Set[str] = set()

        # Dependency graph
        self.dependents: Dict[str, Set[str]] = defaultdict(set)  # task -> tasks that depend on it

        # Worker threads
        self.workers: List[threading.Thread] = []
        self.running = False
        self.lock = threading.Lock()

        # Statistics
        self.stats = {
            'total_tasks': 0,
            'completed': 0,
            'failed': 0,
            'cancelled': 0,
            'total_runtime': 0.0
        }

    def add_task(
        self,
        task_id: str,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        dependencies: Set[str] = None,
        deadline: Optional[datetime] = None,
        estimate_duration: float = 0.0
    ) -> Task:
        """
        Add a task to the scheduler.

        Args:
            task_id: Unique task identifier
            func: Function to execute
            args: Positional arguments
            kwargs: Keyword arguments
            priority: Task priority
            dependencies: Set of task IDs this task depends on
            deadline: Optional deadline
            estimate_duration: Estimated execution time in seconds

        Returns:
            Created Task object
        """
        if task_id in self.tasks:
            raise ValueError(f"Task {task_id} already exists")

        kwargs = kwargs or {}
        dependencies = dependencies or set()

        # Validate dependencies exist
        for dep_id in dependencies:
            if dep_id not in self.tasks and dep_id not in self.completed_tasks:
                logger.warning(f"Task {task_id} depends on unknown task {dep_id}")

        task = Task(
            id=task_id,
            name=func.__name__,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            dependencies=dependencies,
            deadline=deadline,
            estimate_duration=estimate_duration
        )

        with self.lock:
            self.tasks[task_id] = task
            self.stats['total_tasks'] += 1

            # Build dependency graph
            for dep_id in dependencies:
                self.dependents[dep_id].add(task_id)

            # Add to ready queue if no dependencies
            if task.is_ready(self.completed_tasks):
                task.status = TaskStatus.READY
                heapq.heappush(self.ready_queue, task)

        return task

    def remove_task(self, task_id: str) -> bool:
        """Cancel and remove a task"""
        if task_id not in self.tasks:
            return False

        with self.lock:
            task = self.tasks[task_id]

            # Can only cancel pending or ready tasks
            if task.status in [TaskStatus.RUNNING, TaskStatus.COMPLETED]:
                logger.warning(f"Cannot cancel task {task_id} in state {task.status}")
                return False

            task.status = TaskStatus.CANCELLED
            self.stats['cancelled'] += 1

            # Remove from ready queue if present
            self.ready_queue = [t for t in self.ready_queue if t.id != task_id]
            heapq.heapify(self.ready_queue)

            return True

    def _check_dependencies(self):
        """Check if any blocked tasks are now ready"""
        with self.lock:
            newly_ready = []

            for task in self.tasks.values():
                if task.status == TaskStatus.PENDING:
                    if task.is_ready(self.completed_tasks):
                        task.status = TaskStatus.READY
                        newly_ready.append(task)

            # Add newly ready tasks to queue
            for task in newly_ready:
                heapq.heappush(self.ready_queue, task)

    def _execute_task(self, task: Task):
        """Execute a single task"""
        try:
            task.status = TaskStatus.RUNNING
            task.started = time.time()

            logger.info(f"Executing task: {task.id} ({task.name})")

            # Execute function
            task.result = task.func(*task.args, **task.kwargs)

            # Mark completed
            task.status = TaskStatus.COMPLETED
            task.completed = time.time()

            with self.lock:
                self.completed_tasks.add(task.id)
                self.running_tasks.discard(task.id)
                self.stats['completed'] += 1
                self.stats['total_runtime'] += task.duration()

            logger.info(
                f"Task completed: {task.id} in {task.duration():.2f}s"
            )

            # Check if this unblocks other tasks
            self._check_dependencies()

        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}")
            task.error = e
            task.retry_count += 1

            # Retry if not exceeded max retries
            if task.retry_count < task.max_retries:
                task.status = TaskStatus.READY
                with self.lock:
                    heapq.heappush(self.ready_queue, task)
                    self.running_tasks.discard(task.id)
                logger.info(f"Retrying task {task.id} ({task.retry_count}/{task.max_retries})")
            else:
                task.status = TaskStatus.FAILED
                with self.lock:
                    self.failed_tasks.add(task.id)
                    self.running_tasks.discard(task.id)
                    self.stats['failed'] += 1

    def _worker_thread(self):
        """Worker thread that executes tasks from queue"""
        while self.running:
            task = None

            with self.lock:
                # Check for ready tasks
                if self.ready_queue and len(self.running_tasks) < self.max_workers:
                    task = heapq.heappop(self.ready_queue)
                    self.running_tasks.add(task.id)

            if task:
                self._execute_task(task)
            else:
                # No tasks ready, sleep briefly
                time.sleep(0.1)

    def start(self):
        """Start the scheduler (spawn worker threads)"""
        if self.running:
            logger.warning("Scheduler already running")
            return

        self.running = True

        # Spawn worker threads
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_thread,
                name=f"TaskWorker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)

        logger.info(f"Task scheduler started with {self.max_workers} workers")

    def stop(self, wait: bool = True):
        """Stop the scheduler"""
        self.running = False

        if wait:
            for worker in self.workers:
                worker.join(timeout=5.0)

        self.workers.clear()
        logger.info("Task scheduler stopped")

    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all tasks to complete.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if all tasks completed, False if timeout
        """
        start = time.time()

        while True:
            with self.lock:
                pending = len([
                    t for t in self.tasks.values()
                    if t.status in [TaskStatus.PENDING, TaskStatus.READY, TaskStatus.RUNNING]
                ])

                if pending == 0:
                    return True

            # Check timeout
            if timeout and (time.time() - start) > timeout:
                return False

            time.sleep(0.1)

    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get status of a specific task"""
        task = self.tasks.get(task_id)
        return task.status if task else None

    def get_dependency_chain(self, task_id: str) -> List[str]:
        """Get full dependency chain for a task"""
        if task_id not in self.tasks:
            return []

        visited = set()
        chain = []

        def dfs(tid: str):
            if tid in visited:
                return
            visited.add(tid)

            task = self.tasks.get(tid)
            if task:
                for dep in task.dependencies:
                    dfs(dep)
                chain.append(tid)

        dfs(task_id)
        return chain

    def get_critical_path(self) -> List[Task]:
        """
        Calculate critical path (longest dependency chain).
        Useful for identifying bottlenecks.
        """
        # Calculate longest path for each task
        longest_path = {}

        def calculate_longest_path(task_id: str) -> float:
            if task_id in longest_path:
                return longest_path[task_id]

            task = self.tasks.get(task_id)
            if not task:
                return 0.0

            if not task.dependencies:
                longest_path[task_id] = task.estimate_duration
                return task.estimate_duration

            max_dep_path = max(
                (calculate_longest_path(dep) for dep in task.dependencies),
                default=0.0
            )

            longest_path[task_id] = max_dep_path + task.estimate_duration
            return longest_path[task_id]

        # Calculate for all tasks
        for task_id in self.tasks:
            calculate_longest_path(task_id)

        # Find critical path
        if not longest_path:
            return []

        # Start from task with longest path
        current_id = max(longest_path, key=longest_path.get)
        critical = []

        while current_id:
            task = self.tasks[current_id]
            critical.append(task)

            # Find dependency with longest path
            if task.dependencies:
                current_id = max(
                    task.dependencies,
                    key=lambda tid: longest_path.get(tid, 0.0)
                )
            else:
                current_id = None

        return list(reversed(critical))

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        with self.lock:
            pending_count = len([
                t for t in self.tasks.values()
                if t.status == TaskStatus.PENDING
            ])
            ready_count = len(self.ready_queue)
            running_count = len(self.running_tasks)

            return {
                **self.stats,
                'pending': pending_count,
                'ready': ready_count,
                'running': running_count,
                'avg_task_duration': (
                    self.stats['total_runtime'] / self.stats['completed']
                    if self.stats['completed'] > 0 else 0.0
                )
            }


# Global scheduler instance
_global_scheduler = None


def get_scheduler(max_workers: int = 4) -> TaskScheduler:
    """Get or create global task scheduler"""
    global _global_scheduler
    if _global_scheduler is None:
        _global_scheduler = TaskScheduler(max_workers=max_workers)
    return _global_scheduler
