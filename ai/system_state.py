"""
System State Tracker for A.L.I.C.E
Monitors system and task state, publishes events
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from enum import Enum
import threading
import psutil
import logging

from .event_bus import get_event_bus, EventType, EventPriority

logger = logging.getLogger(__name__)


class SystemStatus(Enum):
    """System status states"""
    IDLE = "idle"
    BUSY = "busy"
    THINKING = "thinking"
    EXECUTING = "executing"
    ERROR = "error"


class TaskStatus(Enum):
    """Task execution states"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskState:
    """State of a task"""
    task_id: str
    name: str
    status: TaskStatus
    progress: float = 0.0  # 0.0 to 1.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Get task duration"""
        if self.started_at:
            end = self.completed_at or datetime.now()
            return end - self.started_at
        return None


class SystemStateTracker:
    """
    Tracks system state and publishes events
    Monitors: CPU, memory, tasks, user activity
    """
    
    def __init__(self):
        self.event_bus = get_event_bus()
        
        # System state
        self._status = SystemStatus.IDLE
        self._last_activity = datetime.now()
        
        # Task tracking
        self._tasks: Dict[str, TaskState] = {}
        self._task_counter = 0
        
        # Resource tracking
        self._cpu_threshold = 80.0  # Percentage
        self._memory_threshold = 85.0  # Percentage
        self._storage_threshold = 90.0  # Percentage
        
        # User activity
        self._user_active = True
        self._inactivity_threshold = timedelta(minutes=5)
        
        self._lock = threading.Lock()
        self._monitoring = False
        self._monitor_thread = None
    
    def start_monitoring(self, interval: int = 30):
        """
        Start background monitoring
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True,
            name="system_monitor"
        )
        self._monitor_thread.start()
        logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("System monitoring stopped")
    
    def _monitor_loop(self, interval: int):
        """Background monitoring loop"""
        import time
        
        while self._monitoring:
            try:
                self._check_resources()
                self._check_user_activity()
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
            
            time.sleep(interval)
    
    def _check_resources(self):
        """Check system resources and emit warnings"""
        try:
            # CPU check
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > self._cpu_threshold:
                self.event_bus.emit(
                    EventType.SYSTEM_WARNING,
                    data={"resource": "cpu", "usage": cpu_percent},
                    priority=EventPriority.NORMAL
                )
            
            # Memory check
            memory = psutil.virtual_memory()
            if memory.percent > self._memory_threshold:
                self.event_bus.emit(
                    EventType.MEMORY_LOW,
                    data={"usage": memory.percent, "available": memory.available},
                    priority=EventPriority.HIGH,
                    requires_notification=True
                )
            
            # Storage check
            disk = psutil.disk_usage('/')
            if disk.percent > self._storage_threshold:
                self.event_bus.emit(
                    EventType.STORAGE_LOW,
                    data={"usage": disk.percent, "free": disk.free},
                    priority=EventPriority.HIGH,
                    requires_notification=True
                )
        except Exception as e:
            logger.error(f"Resource check error: {e}")
    
    def _check_user_activity(self):
        """Check user activity and emit idle events"""
        inactive_time = datetime.now() - self._last_activity
        
        if inactive_time > self._inactivity_threshold and self._user_active:
            self._user_active = False
            self.event_bus.emit(
                EventType.USER_INACTIVE,
                data={"inactive_duration": inactive_time.total_seconds()},
                priority=EventPriority.LOW
            )
    
    def mark_user_active(self):
        """Mark user as active"""
        was_inactive = not self._user_active
        self._last_activity = datetime.now()
        self._user_active = True
        
        if was_inactive:
            self.event_bus.emit(
                EventType.USER_ACTIVE,
                priority=EventPriority.LOW
            )
    
    def set_status(self, status: SystemStatus):
        """Update system status"""
        old_status = self._status
        self._status = status
        
        # Emit status change events
        if status == SystemStatus.IDLE and old_status != SystemStatus.IDLE:
            self.event_bus.emit(EventType.SYSTEM_IDLE, priority=EventPriority.LOW)
        elif status == SystemStatus.BUSY and old_status == SystemStatus.IDLE:
            self.event_bus.emit(EventType.SYSTEM_BUSY, priority=EventPriority.LOW)
    
    def get_status(self) -> SystemStatus:
        """Get current system status"""
        return self._status
    
    def create_task(self, name: str, metadata: Dict[str, Any] = None) -> str:
        """
        Create a new task
        
        Args:
            name: Task name
            metadata: Optional task metadata
        
        Returns:
            Task ID
        """
        with self._lock:
            self._task_counter += 1
            task_id = f"task_{self._task_counter}"
            
            task = TaskState(
                task_id=task_id,
                name=name,
                status=TaskStatus.PENDING,
                metadata=metadata or {}
            )
            
            self._tasks[task_id] = task
            
        return task_id
    
    def start_task(self, task_id: str):
        """Start a task"""
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now()
                
                self.event_bus.emit(
                    EventType.TASK_STARTED,
                    data={
                        "task_id": task_id,
                        "name": task.name,
                        "metadata": task.metadata
                    },
                    priority=EventPriority.NORMAL
                )
                
                self.set_status(SystemStatus.EXECUTING)
    
    def update_task_progress(self, task_id: str, progress: float):
        """Update task progress (0.0 to 1.0)"""
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task.progress = max(0.0, min(1.0, progress))
                
                self.event_bus.emit(
                    EventType.TASK_PROGRESS,
                    data={
                        "task_id": task_id,
                        "name": task.name,
                        "progress": task.progress
                    },
                    priority=EventPriority.LOW
                )
    
    def complete_task(self, task_id: str, result: Any = None):
        """Mark task as completed"""
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                task.progress = 1.0
                
                if result:
                    task.metadata['result'] = result
                
                self.event_bus.emit(
                    EventType.TASK_COMPLETED,
                    data={
                        "task_id": task_id,
                        "name": task.name,
                        "duration": task.duration.total_seconds() if task.duration else 0,
                        "metadata": task.metadata
                    },
                    priority=EventPriority.NORMAL
                )
                
                # Check if all tasks are done
                if all(t.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED] 
                       for t in self._tasks.values()):
                    self.set_status(SystemStatus.IDLE)
    
    def fail_task(self, task_id: str, error: str):
        """Mark task as failed"""
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now()
                task.error = error
                
                self.event_bus.emit(
                    EventType.TASK_FAILED,
                    data={
                        "task_id": task_id,
                        "name": task.name,
                        "error": error,
                        "duration": task.duration.total_seconds() if task.duration else 0
                    },
                    priority=EventPriority.HIGH,
                    requires_notification=True
                )
                
                # Check if all tasks are done
                if all(t.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED] 
                       for t in self._tasks.values()):
                    self.set_status(SystemStatus.IDLE)
    
    def get_task(self, task_id: str) -> Optional[TaskState]:
        """Get task state"""
        with self._lock:
            return self._tasks.get(task_id)
    
    def get_active_tasks(self) -> List[TaskState]:
        """Get all active tasks"""
        with self._lock:
            return [
                task for task in self._tasks.values()
                if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]
            ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system state statistics"""
        with self._lock:
            total = len(self._tasks)
            by_status = {}
            
            for task in self._tasks.values():
                status = task.status.value
                by_status[status] = by_status.get(status, 0) + 1
            
            return {
                "status": self._status.value,
                "user_active": self._user_active,
                "total_tasks": total,
                "active_tasks": len(self.get_active_tasks()),
                "tasks_by_status": by_status,
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent
            }


# Global instance
_state_tracker = None


def get_state_tracker() -> SystemStateTracker:
    """Get the global state tracker instance"""
    global _state_tracker
    if _state_tracker is None:
        _state_tracker = SystemStateTracker()
    return _state_tracker
