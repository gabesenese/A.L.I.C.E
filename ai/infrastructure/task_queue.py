"""
Production-Grade Async Task Queue
Supports Celery/RabbitMQ with fallback to in-process threading
"""

import logging
import threading
import queue
import time
from typing import Callable, Any, Optional, Dict
from dataclasses import dataclass
from enum import Enum
from functools import wraps

logger = logging.getLogger(__name__)

# Try to import Celery
try:
    from celery import Celery
    from celery.result import AsyncResult

    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    logger.warning("Celery not available, using in-process task queue")


class TaskPriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Task:
    """Task definition"""

    id: str
    func: Callable
    args: tuple
    kwargs: dict
    priority: TaskPriority = TaskPriority.NORMAL
    retry_count: int = 0
    max_retries: int = 3


class TaskQueue:
    """
    Async task queue for background processing:
    -Email syncing
    - Document indexing
    - Learning model training
    - Long-running API calls
    """

    def __init__(
        self,
        broker_url: str = "redis://localhost:6379/0",
        backend_url: str = "redis://localhost:6379/1",
        enable_celery: bool = True,
        num_workers: int = 4,
    ):
        self.broker_url = broker_url
        self.backend_url = backend_url
        self.num_workers = num_workers
        self.celery_app = None

        # Fallback in-process queue
        self.task_queue = queue.PriorityQueue()
        self.workers = []
        self.running = False
        self.lock = threading.Lock()

        # Task registry
        self.registered_tasks = {}

        # Statistics
        self.stats = {"enqueued": 0, "completed": 0, "failed": 0, "retried": 0}

        if enable_celery and CELERY_AVAILABLE:
            self._initialize_celery()
        else:
            self._initialize_thread_workers()

    def _initialize_celery(self):
        """Initialize Celery for distributed task processing"""
        try:
            self.celery_app = Celery(
                "alice_tasks", broker=self.broker_url, backend=self.backend_url
            )

            self.celery_app.conf.update(
                task_serializer="json",
                accept_content=["json"],
                result_serializer="json",
                timezone="UTC",
                enable_utc=True,
                task_track_started=True,
                task_time_limit=300,  # 5 minutes max
                worker_prefetch_multiplier=1,
                worker_max_tasks_per_child=100,
            )

            logger.info(
                f"[TaskQueue] Celery initialized with broker: {self.broker_url}"
            )

        except Exception as e:
            logger.warning(f"[TaskQueue] Celery init failed: {e}, using thread workers")
            self.celery_app = None
            self._initialize_thread_workers()

    def _initialize_thread_workers(self):
        """Initialize in-process thread workers as fallback"""
        self.running = True

        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop, name=f"TaskWorker-{i}", daemon=True
            )
            worker.start()
            self.workers.append(worker)

        logger.info(f"[TaskQueue] Started {self.num_workers} thread workers")

    def _worker_loop(self):
        """Worker thread loop for processing tasks"""
        while self.running:
            try:
                # Get task with timeout
                priority, task_data = self.task_queue.get(timeout=1)
                task = Task(**task_data)

                logger.debug(f"[TaskQueue] Worker processing task {task.id}")

                try:
                    # Execute task
                    task.func(*task.args, **task.kwargs)

                    with self.lock:
                        self.stats["completed"] += 1

                    logger.info(f"[TaskQueue] Task {task.id} completed")

                except Exception as e:
                    logger.error(f"[TaskQueue] Task {task.id} failed: {e}")

                    # Retry logic
                    if task.retry_count < task.max_retries:
                        task.retry_count += 1
                        self._enqueue_task(task)

                        with self.lock:
                            self.stats["retried"] += 1

                        logger.info(
                            f"[TaskQueue] Retrying task {task.id} ({task.retry_count}/{task.max_retries})"
                        )
                    else:
                        with self.lock:
                            self.stats["failed"] += 1

                finally:
                    self.task_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[TaskQueue] Worker error: {e}")

    def _enqueue_task(self, task: Task):
        """Internal method to enqueue task"""
        priority_value = (
            -task.priority.value
        )  # Negative for min-heap to act as max-heap
        task_data = {
            "id": task.id,
            "func": task.func,
            "args": task.args,
            "kwargs": task.kwargs,
            "priority": task.priority,
            "retry_count": task.retry_count,
            "max_retries": task.max_retries,
        }
        self.task_queue.put((priority_value, task_data))

    def enqueue(
        self,
        func: Callable,
        *args,
        priority: TaskPriority = TaskPriority.NORMAL,
        max_retries: int = 3,
        **kwargs,
    ) -> str:
        """
        Enqueue task for async execution

        Args:
            func: Function to execute
            priority: Task priority
            max_retries: Max retry attempts
            *args, **kwargs: Function arguments

        Returns:
            Task ID
        """
        task_id = f"task_{time.time()}_{id(func)}"

        if self.celery_app:
            # Use Celery
            try:
                task_name = func.__name__
                if task_name in self.registered_tasks:
                    celery_task = self.registered_tasks[task_name]
                    result = celery_task.apply_async(
                        args=args,
                        kwargs=kwargs,
                        priority=priority.value,
                        retry=max_retries > 0,
                        max_retries=max_retries,
                    )
                    task_id = result.id
                else:
                    logger.warning(
                        f"[TaskQueue] Task {task_name} not registered with Celery, using thread pool"
                    )
                    task = Task(
                        id=task_id,
                        func=func,
                        args=args,
                        kwargs=kwargs,
                        priority=priority,
                        max_retries=max_retries,
                    )
                    self._enqueue_task(task)
            except Exception as e:
                logger.warning(
                    f"[TaskQueue] Celery enqueue failed: {e}, using thread pool"
                )
                task = Task(
                    id=task_id,
                    func=func,
                    args=args,
                    kwargs=kwargs,
                    priority=priority,
                    max_retries=max_retries,
                )
                self._enqueue_task(task)
        else:
            # Use thread pool
            task = Task(
                id=task_id,
                func=func,
                args=args,
                kwargs=kwargs,
                priority=priority,
                max_retries=max_retries,
            )
            self._enqueue_task(task)

        with self.lock:
            self.stats["enqueued"] += 1

        logger.debug(f"[TaskQueue] Enqueued task {task_id}")
        return task_id

    def register_task(self, name: str = None):
        """
        Decorator to register task with Celery

        Usage:
            @task_queue.register_task()
            def process_emails():
                ...
        """

        def decorator(func: Callable) -> Callable:
            task_name = name or func.__name__

            if self.celery_app:
                # Register with Celery
                celery_task = self.celery_app.task(name=task_name, bind=True)(func)
                self.registered_tasks[task_name] = celery_task
                logger.debug(f"[TaskQueue] Registered Celery task: {task_name}")
            else:
                # Just track for thread pool
                self.registered_tasks[task_name] = func

            @wraps(func)
            def wrapper(*args, **kwargs):
                # Allow calling directly or via queue
                return func(*args, **kwargs)

            # Add async method
            wrapper.delay = lambda *args, **kwargs: self.enqueue(func, *args, **kwargs)

            return wrapper

        return decorator

    def get_task_status(self, task_id: str) -> Optional[str]:
        """Get status of a task"""
        if self.celery_app:
            try:
                result = AsyncResult(task_id, app=self.celery_app)
                return result.state
            except:
                return None
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        with self.lock:
            stats = self.stats.copy()

        stats["queue_size"] = self.task_queue.qsize()
        stats["workers"] = len(self.workers)
        stats["backend"] = "celery" if self.celery_app else "threads"

        return stats

    def stop(self):
        """Stop all workers"""
        self.running = False

        for worker in self.workers:
            worker.join(timeout=5)

        logger.info("[TaskQueue] All workers stopped")


# Global task queue instance
_task_queue = None


def get_task_queue() -> TaskQueue:
    """Get global task queue instance"""
    global _task_queue
    if _task_queue is None:
        _task_queue = TaskQueue()
    return _task_queue


def initialize_task_queue(
    broker_url="redis://localhost:6379/0",
    backend_url="redis://localhost:6379/1",
    num_workers: int = 4,
) -> TaskQueue:
    """Initialize global task queue"""
    global _task_queue
    _task_queue = TaskQueue(
        broker_url=broker_url, backend_url=backend_url, num_workers=num_workers
    )
    return _task_queue


# Example background tasks
def example_email_sync():
    """Example: Sync emails in background"""
    logger.info("[Task] Starting email sync...")
    time.sleep(5)  # Simulate work
    logger.info("[Task] Email sync completed")
    return {"synced": 42, "new": 5}


def example_document_index(file_path: str):
    """Example: Index document for search"""
    logger.info(f"[Task] Indexing {file_path}...")
    time.sleep(2)  # Simulate work
    logger.info(f"[Task] Indexed {file_path}")
    return {"indexed": file_path, "words": 1234}
