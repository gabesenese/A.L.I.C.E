"""
Background Observers for A.L.I.C.E
Smart observers that watch events and decide when to notify
"""

from typing import Optional, Callable, List
from datetime import datetime, timedelta
import logging

from .event_bus import get_event_bus, Event, EventType, EventPriority

logger = logging.getLogger(__name__)


class Observer:
    """
    Base observer class
    Watches events and decides when to interrupt/notify
    """
    
    def __init__(self, name: str):
        self.name = name
        self.event_bus = get_event_bus()
        self.enabled = True
        self._on_notification: Optional[Callable] = None
    
    def set_notification_callback(self, callback: Callable):
        """Set callback for when observer wants to notify user"""
        self._on_notification = callback
    
    def notify(self, message: str, priority: EventPriority = EventPriority.NORMAL):
        """Trigger a notification"""
        if self._on_notification and self.enabled:
            logger.info(f"Observer '{self.name}' notification: {message}")
            self._on_notification(message, priority)
    
    def start(self):
        """Start observing (subscribe to events)"""
        raise NotImplementedError
    
    def stop(self):
        """Stop observing (unsubscribe from events)"""
        raise NotImplementedError


class TaskObserver(Observer):
    """
    Watches task execution
    Notifies on: failures, long-running tasks, important completions
    """
    
    def __init__(self):
        super().__init__("TaskObserver")
        self._task_start_times = {}
        self._long_running_threshold = timedelta(minutes=2)
        self._notified_tasks = set()
    
    def start(self):
        """Start observing task events"""
        self.event_bus.subscribe(EventType.TASK_STARTED, self._on_task_started)
        self.event_bus.subscribe(EventType.TASK_COMPLETED, self._on_task_completed)
        self.event_bus.subscribe(EventType.TASK_FAILED, self._on_task_failed)
        self.event_bus.subscribe(EventType.TASK_PROGRESS, self._on_task_progress)
        logger.info("TaskObserver started")
    
    def stop(self):
        """Stop observing"""
        self.event_bus.unsubscribe(EventType.TASK_STARTED, self._on_task_started)
        self.event_bus.unsubscribe(EventType.TASK_COMPLETED, self._on_task_completed)
        self.event_bus.unsubscribe(EventType.TASK_FAILED, self._on_task_failed)
        self.event_bus.unsubscribe(EventType.TASK_PROGRESS, self._on_task_progress)
    
    def _on_task_started(self, event: Event):
        """Track task start"""
        task_id = event.data.get('task_id')
        if task_id:
            self._task_start_times[task_id] = datetime.now()
    
    def _on_task_completed(self, event: Event):
        """Check if completion is worth notifying"""
        task_id = event.data.get('task_id')
        duration = event.data.get('duration', 0)
        
        # Notify if task took longer than threshold
        if duration > self._long_running_threshold.total_seconds():
            self.notify(
                f"Task '{event.data.get('name')}' completed after {duration:.1f}s",
                priority=EventPriority.NORMAL
            )
        
        # Clean up
        if task_id in self._task_start_times:
            del self._task_start_times[task_id]
        if task_id in self._notified_tasks:
            self._notified_tasks.remove(task_id)
    
    def _on_task_failed(self, event: Event):
        """Always notify on task failure"""
        task_name = event.data.get('name', 'Unknown task')
        error = event.data.get('error', 'Unknown error')
        
        self.notify(
            f"Task '{task_name}' failed: {error}",
            priority=EventPriority.HIGH
        )
    
    def _on_task_progress(self, event: Event):
        """Check for long-running tasks"""
        task_id = event.data.get('task_id')
        
        if task_id in self._task_start_times and task_id not in self._notified_tasks:
            elapsed = datetime.now() - self._task_start_times[task_id]
            
            if elapsed > self._long_running_threshold:
                progress = event.data.get('progress', 0)
                task_name = event.data.get('name', 'Task')
                
                self.notify(
                    f"'{task_name}' is taking longer than expected ({int(progress * 100)}% done)",
                    priority=EventPriority.LOW
                )
                
                self._notified_tasks.add(task_id)


class SystemHealthObserver(Observer):
    """
    Watches system health
    Notifies on: low memory, low storage, high CPU
    """
    
    def __init__(self):
        super().__init__("SystemHealthObserver")
        self._last_memory_warning = None
        self._warning_cooldown = timedelta(minutes=10)
    
    def start(self):
        """Start observing system events"""
        self.event_bus.subscribe(EventType.MEMORY_LOW, self._on_memory_low)
        self.event_bus.subscribe(EventType.STORAGE_LOW, self._on_storage_low)
        self.event_bus.subscribe(EventType.SYSTEM_ERROR, self._on_system_error)
        logger.info("SystemHealthObserver started")
    
    def stop(self):
        """Stop observing"""
        self.event_bus.unsubscribe(EventType.MEMORY_LOW, self._on_memory_low)
        self.event_bus.unsubscribe(EventType.STORAGE_LOW, self._on_storage_low)
        self.event_bus.unsubscribe(EventType.SYSTEM_ERROR, self._on_system_error)
    
    def _on_memory_low(self, event: Event):
        """Notify on low memory (with cooldown)"""
        now = datetime.now()
        
        if self._last_memory_warning is None or \
           (now - self._last_memory_warning) > self._warning_cooldown:
            
            usage = event.data.get('usage', 0)
            available = event.data.get('available', 0)
            available_gb = available / (1024**3)
            
            self.notify(
                f"Memory usage high ({usage:.1f}%). {available_gb:.1f}GB available.",
                priority=EventPriority.HIGH
            )
            
            self._last_memory_warning = now
    
    def _on_storage_low(self, event: Event):
        """Notify on low storage"""
        usage = event.data.get('usage', 0)
        free = event.data.get('free', 0)
        free_gb = free / (1024**3)
        
        self.notify(
            f"Storage space low ({usage:.1f}%). {free_gb:.1f}GB free.",
            priority=EventPriority.HIGH
        )
    
    def _on_system_error(self, event: Event):
        """Notify on system errors"""
        error = event.data.get('error', 'Unknown error')
        
        self.notify(
            f"System error: {error}",
            priority=EventPriority.CRITICAL
        )


class ReminderObserver(Observer):
    """
    Watches for reminders and calendar events
    Notifies when events are due or approaching
    """
    
    def __init__(self):
        super().__init__("ReminderObserver")
    
    def start(self):
        """Start observing reminder events"""
        self.event_bus.subscribe(EventType.REMINDER_DUE, self._on_reminder_due)
        self.event_bus.subscribe(EventType.EVENT_APPROACHING, self._on_event_approaching)
        logger.info("ReminderObserver started")
    
    def stop(self):
        """Stop observing"""
        self.event_bus.unsubscribe(EventType.REMINDER_DUE, self._on_reminder_due)
        self.event_bus.unsubscribe(EventType.EVENT_APPROACHING, self._on_event_approaching)
    
    def _on_reminder_due(self, event: Event):
        """Notify on due reminders"""
        title = event.data.get('title', 'Reminder')
        
        self.notify(
            f"Reminder: {title}",
            priority=EventPriority.HIGH
        )
    
    def _on_event_approaching(self, event: Event):
        """Notify on approaching events"""
        title = event.data.get('title', 'Event')
        time_until = event.data.get('time_until', 'soon')
        
        self.notify(
            f"Upcoming: {title} in {time_until}",
            priority=EventPriority.NORMAL
        )


class BackgroundActivityObserver(Observer):
    """
    Watches background activity
    Notifies on: important emails, file changes, calendar updates
    """
    
    def __init__(self):
        super().__init__("BackgroundActivityObserver")
        self._important_senders = []  # Could be loaded from config
    
    def start(self):
        """Start observing background events"""
        self.event_bus.subscribe(EventType.EMAIL_RECEIVED, self._on_email_received)
        self.event_bus.subscribe(EventType.CALENDAR_UPDATED, self._on_calendar_updated)
        self.event_bus.subscribe(EventType.FILE_CHANGED, self._on_file_changed)
        logger.info("BackgroundActivityObserver started")
    
    def stop(self):
        """Stop observing"""
        self.event_bus.unsubscribe(EventType.EMAIL_RECEIVED, self._on_email_received)
        self.event_bus.unsubscribe(EventType.CALENDAR_UPDATED, self._on_calendar_updated)
        self.event_bus.unsubscribe(EventType.FILE_CHANGED, self._on_file_changed)
    
    def _on_email_received(self, event: Event):
        """Notify on important emails only"""
        sender = event.data.get('sender', '')
        subject = event.data.get('subject', '')
        is_important = event.data.get('important', False)
        
        # Only notify if flagged as important or from important sender
        if is_important or any(s in sender for s in self._important_senders):
            self.notify(
                f"New email from {sender}: {subject}",
                priority=EventPriority.NORMAL
            )
    
    def _on_calendar_updated(self, event: Event):
        """Notify on calendar changes"""
        change_type = event.data.get('type', 'updated')
        event_title = event.data.get('title', 'Event')
        
        # Only notify on new events or cancellations
        if change_type in ['created', 'cancelled']:
            self.notify(
                f"Calendar {change_type}: {event_title}",
                priority=EventPriority.LOW
            )
    
    def _on_file_changed(self, event: Event):
        """Watch for important file changes (configurable)"""
        # This could be extended to watch specific files
        # For now, we don't notify unless explicitly marked
        if event.data.get('notify', False):
            filepath = event.data.get('path', 'File')
            self.notify(
                f"File changed: {filepath}",
                priority=EventPriority.LOW
            )


class ObserverManager:
    """Manages all observers"""
    
    def __init__(self):
        self.observers: List[Observer] = []
        self._notification_callback: Optional[Callable] = None
    
    def register(self, observer: Observer):
        """Register an observer"""
        if observer not in self.observers:
            observer.set_notification_callback(self._handle_notification)
            self.observers.append(observer)
            logger.info(f"Registered observer: {observer.name}")
    
    def set_notification_callback(self, callback: Callable):
        """Set the callback for when observers want to notify"""
        self._notification_callback = callback
    
    def _handle_notification(self, message: str, priority: EventPriority):
        """Handle notification from an observer"""
        if self._notification_callback:
            self._notification_callback(message, priority)
    
    def start_all(self):
        """Start all observers"""
        for observer in self.observers:
            try:
                observer.start()
            except Exception as e:
                logger.error(f"Failed to start {observer.name}: {e}")
    
    def stop_all(self):
        """Stop all observers"""
        for observer in self.observers:
            try:
                observer.stop()
            except Exception as e:
                logger.error(f"Failed to stop {observer.name}: {e}")
    
    def enable_observer(self, name: str):
        """Enable a specific observer"""
        for observer in self.observers:
            if observer.name == name:
                observer.enabled = True
                logger.info(f"Enabled observer: {name}")
                return
    
    def disable_observer(self, name: str):
        """Disable a specific observer"""
        for observer in self.observers:
            if observer.name == name:
                observer.enabled = False
                logger.info(f"Disabled observer: {name}")
                return


# Global instance
_observer_manager = None


def get_observer_manager() -> ObserverManager:
    """Get the global observer manager"""
    global _observer_manager
    if _observer_manager is None:
        _observer_manager = ObserverManager()
        
        # Register default observers
        _observer_manager.register(TaskObserver())
        _observer_manager.register(SystemHealthObserver())
        _observer_manager.register(ReminderObserver())
        _observer_manager.register(BackgroundActivityObserver())
    
    return _observer_manager
