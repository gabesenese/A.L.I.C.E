"""
Example: Integrating Event System into A.L.I.C.E
Shows how to use event bus, state tracker, and observers
"""

from ai.event_bus import get_event_bus, EventType, EventPriority
from ai.system_state import get_state_tracker, SystemStatus
from ai.observers import get_observer_manager


def example_task_tracking():
    """Example: Track a task with events"""
    state = get_state_tracker()
    
    # Create a task
    task_id = state.create_task("Download large file", metadata={"size": "5GB"})
    
    # Start the task
    state.start_task(task_id)
    
    # Update progress
    state.update_task_progress(task_id, 0.5)
    
    # Complete the task
    state.complete_task(task_id, result="Success")


def example_custom_events():
    """Example: Publish custom events"""
    bus = get_event_bus()
    
    # Emit a custom event
    bus.emit(
        EventType.PLUGIN_NOTIFICATION,
        data={
            "plugin": "email",
            "message": "5 new emails received"
        },
        priority=EventPriority.NORMAL,
        requires_notification=True
    )


def example_observers():
    """Example: Set up observers with notification callback"""
    manager = get_observer_manager()
    
    # Define how ALICE should notify user
    def handle_notification(message: str, priority: EventPriority):
        # In real implementation, this would:
        # - Display in terminal (Rich UI)
        # - Speak message (if voice enabled)
        # - Log to conversation
        print(f"[{priority.name}] {message}")
    
    # Set notification callback
    manager.set_notification_callback(handle_notification)
    
    # Start observing
    manager.start_all()
    
    # Now observers will watch events and notify when appropriate
    # They decide WHEN to interrupt based on:
    # - Event priority
    # - User activity state
    # - Notification cooldowns
    # - Task importance


def example_resource_monitoring():
    """Example: Start system resource monitoring"""
    state = get_state_tracker()
    
    # Start monitoring (checks every 30 seconds)
    state.start_monitoring(interval=30)
    
    # Observer will automatically notify on:
    # - High memory usage (>85%)
    # - Low storage (<10% free)
    # - User inactivity (>5 min)


if __name__ == "__main__":
    print("Event System Examples")
    print("=" * 50)
    
    # Set up observers
    example_observers()
    
    # Start monitoring
    example_resource_monitoring()
    
    # Simulate task
    example_task_tracking()
    
    # Emit custom event
    example_custom_events()
    
    # Check event bus stats
    bus = get_event_bus()
    print("\nEvent Bus Stats:")
    print(bus.get_stats())
    
    # Check system state
    state = get_state_tracker()
    print("\nSystem State:")
    print(state.get_stats())
