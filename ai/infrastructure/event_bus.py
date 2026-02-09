"""
Event Bus System for A.L.I.C.E
Lightweight publish-subscribe pattern for system events
"""

from typing import Callable, Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import threading
import logging

logger = logging.getLogger(__name__)


class EventPriority(Enum):
    """Event priority levels"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class EventType(Enum):
    """System event types"""
    # Task events
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_PROGRESS = "task_progress"
    
    # System events
    SYSTEM_IDLE = "system_idle"
    SYSTEM_BUSY = "system_busy"
    SYSTEM_ERROR = "system_error"
    SYSTEM_WARNING = "system_warning"
    
    # Memory events
    MEMORY_LOW = "memory_low"
    STORAGE_LOW = "storage_low"
    
    # Plugin events
    PLUGIN_LOADED = "plugin_loaded"
    PLUGIN_FAILED = "plugin_failed"
    PLUGIN_NOTIFICATION = "plugin_notification"
    
    # User events
    USER_INACTIVE = "user_inactive"
    USER_ACTIVE = "user_active"
    USER_QUERY = "user_query"
    
    # Calendar/Reminder events
    REMINDER_DUE = "reminder_due"
    EVENT_APPROACHING = "event_approaching"
    
    # Background events
    FILE_CHANGED = "file_changed"
    EMAIL_RECEIVED = "email_received"
    CALENDAR_UPDATED = "calendar_updated"


@dataclass
class Event:
    """Event data structure"""
    event_type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    priority: EventPriority = EventPriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "system"
    requires_notification: bool = False
    
    def __repr__(self):
        return f"Event({self.event_type.value}, priority={self.priority.name}, notify={self.requires_notification})"


class EventBus:
    """
    Lightweight event bus for system-wide events
    Allows components to communicate without tight coupling
    """
    
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._event_history: List[Event] = []
        self._max_history = 1000
        self._lock = threading.Lock()
        
    def subscribe(self, event_type: EventType, callback: Callable):
        """
        Subscribe to an event type
        
        Args:
            event_type: Type of event to listen for
            callback: Function to call when event occurs (receives Event object)
        """
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            
            if callback not in self._subscribers[event_type]:
                self._subscribers[event_type].append(callback)
                logger.debug(f"Subscribed {callback.__name__} to {event_type.value}")
    
    def unsubscribe(self, event_type: EventType, callback: Callable):
        """Unsubscribe from an event type"""
        with self._lock:
            if event_type in self._subscribers:
                if callback in self._subscribers[event_type]:
                    self._subscribers[event_type].remove(callback)
                    logger.debug(f"Unsubscribed {callback.__name__} from {event_type.value}")
    
    def publish(self, event: Event):
        """
        Publish an event to all subscribers
        
        Args:
            event: Event to publish
        """
        # Store in history
        with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)
        
        # Notify subscribers
        subscribers = self._subscribers.get(event.event_type, [])
        
        logger.debug(f"Publishing {event} to {len(subscribers)} subscribers")
        
        for callback in subscribers:
            try:
                # Run callback in separate thread for non-blocking
                threading.Thread(
                    target=callback,
                    args=(event,),
                    daemon=True,
                    name=f"event_{event.event_type.value}"
                ).start()
            except Exception as e:
                logger.error(f"Error in event callback {callback.__name__}: {e}")
    
    def emit(self, event_type: EventType, data: Dict[str, Any] = None, 
             priority: EventPriority = EventPriority.NORMAL,
             requires_notification: bool = False,
             source: str = "system"):
        """
        Convenient method to create and publish an event
        
        Args:
            event_type: Type of event
            data: Event data
            priority: Event priority
            requires_notification: Whether to notify user
            source: Event source identifier
        """
        event = Event(
            event_type=event_type,
            data=data or {},
            priority=priority,
            requires_notification=requires_notification,
            source=source
        )
        self.publish(event)
    
    def get_history(self, event_type: EventType = None, limit: int = 100) -> List[Event]:
        """
        Get event history
        
        Args:
            event_type: Filter by event type (None for all)
            limit: Maximum number of events to return
        
        Returns:
            List of events
        """
        with self._lock:
            if event_type:
                events = [e for e in self._event_history if e.event_type == event_type]
            else:
                events = self._event_history.copy()
            
            return events[-limit:]
    
    def clear_history(self):
        """Clear event history"""
        with self._lock:
            self._event_history.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics"""
        with self._lock:
            total_subscribers = sum(len(subs) for subs in self._subscribers.values())
            event_counts = {}
            
            for event in self._event_history:
                event_type = event.event_type.value
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            return {
                "total_subscribers": total_subscribers,
                "total_events": len(self._event_history),
                "event_types": len(self._subscribers),
                "event_counts": event_counts
            }
    
    def emit_custom(self, event_name: str, data: Dict[str, Any] = None, priority: EventPriority = EventPriority.NORMAL):
        """
        Emit a custom named event (for routing, metrics, etc.)
        
        Args:
            event_name: Custom event name (e.g., 'routing.self_reflection')
            data: Event data
            priority: Event priority
        """
        # Call subscribers if any registered for this custom event
        with self._lock:
            subscribers = self._subscribers.get(event_name, [])
        
        for callback in subscribers:
            try:
                event = Event(
                    event_type=EventType.USER_QUERY,  # Use generic type
                    data={'event_name': event_name, **(data or {})},
                    priority=priority
                )
                callback(event)
            except Exception as e:
                logger.error(f"Error calling subscriber for custom event {event_name}: {e}")
    
    def subscribe_to_custom(self, event_name: str, callback: Callable):
        """
        Subscribe to custom events
        
        Args:
            event_name: Custom event name pattern
            callback: Function to call when event occurs
        """
        with self._lock:
            if event_name not in self._subscribers:
                self._subscribers[event_name] = []
            
            if callback not in self._subscribers[event_name]:
                self._subscribers[event_name].append(callback)
    
    def emit_routing_event(self, stage: str, data: Dict[str, Any]):
        """
        Emit a routing event for metrics collection
        
        Args:
            stage: Routing stage (e.g., 'self_reflection', 'conversational', 'tool', 'rag', 'llm')
            data: Routing decision data
        """
        self.emit_custom(f'routing.{stage}', data, priority=EventPriority.HIGH)


# Global event bus instance
_event_bus = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance"""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus
