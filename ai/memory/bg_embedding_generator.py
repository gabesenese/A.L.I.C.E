"""
Background Embedding Generator
Generates embeddings asynchronously in background to avoid blocking main thread
"""

import logging
import threading
import queue
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingTask:
    """Represents an embedding generation task"""
    task_id: str
    text: str
    callback: Optional[Callable] = None
    metadata: Optional[Dict[str, Any]] = None
    priority: int = 0  # Higher priority = processed first
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class BackgroundEmbeddingGenerator:
    """
    Generates embeddings in a background thread
    - Non-blocking API for embedding generation
    - Priority queue for important tasks
    - Batch processing for efficiency
    - Callback support for async completion
    """

    def __init__(self, embedding_manager, max_queue_size: int = 1000, batch_size: int = 10):
        self.embedding_manager = embedding_manager
        self.max_queue_size = max_queue_size
        self.batch_size = batch_size

        # Task queue (priority queue)
        self.task_queue = queue.PriorityQueue(maxsize=max_queue_size)

        # Results cache (task_id -> embedding)
        self.results_cache: Dict[str, Any] = {}
        self.cache_lock = threading.Lock()

        # Worker thread
        self.worker_thread = None
        self.running = False

        # Statistics
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'queue_full_count': 0,
            'total_processing_time': 0.0
        }
        self.stats_lock = threading.Lock()

    def start(self):
        """Start the background worker thread"""
        if self.running:
            logger.warning("[BgEmbedding] Already running")
            return

        self.running = True
        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="EmbeddingWorker"
        )
        self.worker_thread.start()
        logger.info("[BgEmbedding] Background embedding generator started")

    def stop(self):
        """Stop the background worker thread"""
        if not self.running:
            return

        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        logger.info("[BgEmbedding] Background embedding generator stopped")

    def submit_task(
        self,
        task_id: str,
        text: str,
        callback: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None,
        priority: int = 0
    ) -> bool:
        """
        Submit an embedding task to the queue
        Returns True if successfully queued, False if queue is full
        """
        if not self.running:
            logger.warning("[BgEmbedding] Generator not running, call start() first")
            return False

        task = EmbeddingTask(
            task_id=task_id,
            text=text,
            callback=callback,
            metadata=metadata,
            priority=priority
        )

        try:
            # Priority queue uses tuple: (priority, task)
            # Negative priority for descending order (higher priority first)
            self.task_queue.put((-priority, task), block=False)

            with self.stats_lock:
                self.stats['total_tasks'] += 1

            logger.debug(f"[BgEmbedding] Queued task {task_id} (priority={priority})")
            return True

        except queue.Full:
            with self.stats_lock:
                self.stats['queue_full_count'] += 1
            logger.warning(f"[BgEmbedding] Queue full, cannot queue task {task_id}")
            return False

    def get_result(self, task_id: str, timeout: float = None) -> Optional[Any]:
        """
        Get embedding result for a task
        If timeout is specified, waits up to that many seconds
        Returns None if not ready or not found
        """
        if timeout:
            wait_end = time.time() + timeout
            while time.time() < wait_end:
                with self.cache_lock:
                    if task_id in self.results_cache:
                        return self.results_cache.pop(task_id)
                time.sleep(0.1)
            return None
        else:
            with self.cache_lock:
                return self.results_cache.pop(task_id, None)

    def is_task_complete(self, task_id: str) -> bool:
        """Check if a task is complete"""
        with self.cache_lock:
            return task_id in self.results_cache

    def _worker_loop(self):
        """Main worker loop - processes tasks from queue"""
        logger.info("[BgEmbedding] Worker loop started")

        batch_tasks = []
        last_batch_time = time.time()
        batch_timeout = 1.0  # Process batch after 1 second even if not full

        while self.running:
            try:
                # Get task from queue (with timeout to check running flag periodically)
                try:
                    priority, task = self.task_queue.get(timeout=0.5)
                    batch_tasks.append(task)
                except queue.Empty:
                    pass

                # Process batch if:
                # 1. Batch is full
                # 2. Batch timeout reached and batch not empty
                # 3. Queue is empty and batch not empty (clean up remaining)
                current_time = time.time()
                should_process = (
                    len(batch_tasks) >= self.batch_size or
                    (batch_tasks and current_time - last_batch_time >= batch_timeout) or
                    (batch_tasks and self.task_queue.empty())
                )

                if should_process:
                    self._process_batch(batch_tasks)
                    batch_tasks = []
                    last_batch_time = time.time()

            except Exception as e:
                logger.error(f"[BgEmbedding] Error in worker loop: {e}", exc_info=True)
                time.sleep(1)  # Avoid tight loop on error

        # Process remaining tasks before exit
        if batch_tasks:
            self._process_batch(batch_tasks)

        logger.info("[BgEmbedding] Worker loop stopped")

    def _process_batch(self, tasks: List[EmbeddingTask]):
        """Process a batch of tasks"""
        if not tasks:
            return

        start_time = time.time()

        try:
            # Extract texts
            texts = [task.text for task in tasks]

            # Generate embeddings in batch
            embeddings = self.embedding_manager.batch_create_embeddings(texts)

            # Store results and call callbacks
            for task, embedding in zip(tasks, embeddings):
                # Store in cache
                with self.cache_lock:
                    self.results_cache[task.task_id] = {
                        'embedding': embedding,
                        'task_id': task.task_id,
                        'metadata': task.metadata,
                        'completed_at': datetime.now().isoformat()
                    }

                # Call callback if provided
                if task.callback:
                    try:
                        task.callback(task.task_id, embedding, task.metadata)
                    except Exception as e:
                        logger.error(f"[BgEmbedding] Callback error for task {task.task_id}: {e}")

                with self.stats_lock:
                    self.stats['completed_tasks'] += 1

            processing_time = time.time() - start_time
            with self.stats_lock:
                self.stats['total_processing_time'] += processing_time

            logger.debug(f"[BgEmbedding] Processed batch of {len(tasks)} tasks in {processing_time:.2f}s")

        except Exception as e:
            logger.error(f"[BgEmbedding] Batch processing error: {e}", exc_info=True)

            with self.stats_lock:
                self.stats['failed_tasks'] += len(tasks)

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        with self.stats_lock:
            stats_copy = self.stats.copy()

        stats_copy['queue_size'] = self.task_queue.qsize()
        stats_copy['cache_size'] = len(self.results_cache)
        stats_copy['running'] = self.running

        if stats_copy['completed_tasks'] > 0:
            stats_copy['avg_processing_time'] = (
                stats_copy['total_processing_time'] / stats_copy['completed_tasks']
            )
        else:
            stats_copy['avg_processing_time'] = 0.0

        return stats_copy

    def clear_cache(self):
        """Clear the results cache"""
        with self.cache_lock:
            cleared_count = len(self.results_cache)
            self.results_cache.clear()

        logger.info(f"[BgEmbedding] Cleared {cleared_count} cached results")

    def get_queue_size(self) -> int:
        """Get current queue size"""
        return self.task_queue.qsize()

    def is_queue_full(self) -> bool:
        """Check if queue is full"""
        return self.task_queue.full()


# Singleton factory
_bg_embedding_generator = None

def get_bg_embedding_generator(embedding_manager, **kwargs) -> BackgroundEmbeddingGenerator:
    """Get singleton background embedding generator"""
    global _bg_embedding_generator
    if _bg_embedding_generator is None:
        _bg_embedding_generator = BackgroundEmbeddingGenerator(embedding_manager, **kwargs)
    return _bg_embedding_generator
