"""
Service Degradation Handler for A.L.I.C.E
Provides graceful fallback when external services are unavailable
"""

import logging
from typing import Optional, Dict, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from ai.errors import (
    OllamaConnectionError, EmailError, CalendarError,
    NetworkError, LLMError, ToolError
)

logger = logging.getLogger(__name__)


@dataclass
class ServiceStatus:
    """Track service availability status"""
    service_name: str
    is_available: bool
    last_check: datetime
    failure_count: int = 0
    last_error: Optional[str] = None
    degraded_mode_message: str = ""


class ServiceDegradationHandler:
    """
    Handle service failures with graceful degradation
    
    Features:
    - Automatic fallback modes
    - Service health tracking
    - User-friendly error messages
    - Retry logic with backoff
    """
    
    def __init__(self):
        self.services: Dict[str, ServiceStatus] = {}
        
        # Check intervals (seconds)
        self.check_interval = 60  # Re-check failed services every 60s
        
        # Failure thresholds
        self.max_failures_before_disable = 3
        
        # Initialize service statuses
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize service status tracking"""
        services = [
            ("ollama", "AI generation unavailable - using cached responses"),
            ("gmail", "Email features unavailable"),
            ("calendar", "Calendar features unavailable"),
            ("weather", "Weather information unavailable"),
            ("music", "Music playback unavailable"),
            ("web_search", "Web search unavailable"),
            ("documents", "Document query unavailable - stored documents not searchable"),
        ]
        
        for service_name, degraded_message in services:
            self.services[service_name] = ServiceStatus(
                service_name=service_name,
                is_available=True,  # Assume available until proven otherwise
                last_check=datetime.now(),
                degraded_mode_message=degraded_message
            )
    
    def mark_service_failed(self, service_name: str, error: Exception):
        """Mark service as failed and update status"""
        if service_name not in self.services:
            logger.warning(f"Unknown service: {service_name}")
            return
        
        service = self.services[service_name]
        service.failure_count += 1
        service.last_error = str(error)
        service.last_check = datetime.now()
        
        if service.failure_count >= self.max_failures_before_disable:
            if service.is_available:
                logger.warning(f"🔴 Service '{service_name}' is now UNAVAILABLE after {service.failure_count} failures")
                service.is_available = False
        else:
            logger.warning(f"⚠️ Service '{service_name}' failed ({service.failure_count}/{self.max_failures_before_disable}): {error}")
    
    def mark_service_recovered(self, service_name: str):
        """Mark service as recovered"""
        if service_name not in self.services:
            return
        
        service = self.services[service_name]
        if not service.is_available:
            logger.info(f"✅ Service '{service_name}' is now AVAILABLE")
        
        service.is_available = True
        service.failure_count = 0
        service.last_error = None
        service.last_check = datetime.now()
    
    def should_retry_service(self, service_name: str) -> bool:
        """Determine if service should be retried"""
        if service_name not in self.services:
            return True
        
        service = self.services[service_name]
        
        # If service is available, always try
        if service.is_available:
            return True
        
        # If service is down, check if enough time has passed to retry
        time_since_check = datetime.now() - service.last_check
        return time_since_check.total_seconds() >= self.check_interval
    
    def get_service_status(self, service_name: str) -> ServiceStatus:
        """Get current service status"""
        return self.services.get(service_name, ServiceStatus(
            service_name=service_name,
            is_available=True,
            last_check=datetime.now()
        ))
    
    def get_all_statuses(self) -> Dict[str, ServiceStatus]:
        """Get all service statuses"""
        return self.services.copy()
    
    def get_degraded_message(self, service_name: str) -> str:
        """Get user-friendly message for degraded service"""
        service = self.get_service_status(service_name)
        return service.degraded_mode_message
    
    def with_fallback(
        self,
        service_name: str,
        primary_func: Callable,
        fallback_func: Optional[Callable] = None,
        fallback_value: Any = None
    ) -> Any:
        """
        Execute function with automatic fallback
        
        Args:
            service_name: Name of service being called
            primary_func: Primary function to execute
            fallback_func: Fallback function if primary fails
            fallback_value: Static fallback value if no fallback_func
        
        Returns:
            Result from primary_func, fallback_func, or fallback_value
        """
        # Check if we should even try
        if not self.should_retry_service(service_name):
            logger.info(f"⏭️ Skipping '{service_name}' (service unavailable)")
            if fallback_func:
                return fallback_func()
            return fallback_value
        
        # Try primary function
        try:
            result = primary_func()
            
            # Mark success
            if not self.get_service_status(service_name).is_available:
                self.mark_service_recovered(service_name)
            
            return result
            
        except (OllamaConnectionError, EmailError, CalendarError, NetworkError, LLMError, ToolError) as e:
            # Mark failure
            self.mark_service_failed(service_name, e)
            
            # Try fallback
            logger.info(f"⚠️ Using fallback for '{service_name}': {e.user_message}")
            
            if fallback_func:
                try:
                    return fallback_func()
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
                    return fallback_value
            
            return fallback_value
        
        except Exception as e:
            # Unexpected error - don't mark service as failed
            logger.error(f"Unexpected error in '{service_name}': {e}")
            raise


# Global singleton
_degradation_handler = None

def get_degradation_handler() -> ServiceDegradationHandler:
    """Get global degradation handler instance"""
    global _degradation_handler
    if _degradation_handler is None:
        _degradation_handler = ServiceDegradationHandler()
    return _degradation_handler


# Convenience decorators
def with_ollama_fallback(fallback_response: str = "I'm having trouble generating a response right now."):
    """Decorator for functions that use Ollama"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            handler = get_degradation_handler()
            return handler.with_fallback(
                "ollama",
                lambda: func(*args, **kwargs),
                fallback_value=fallback_response
            )
        return wrapper
    return decorator


def with_email_fallback(fallback_response: str = "Email service is temporarily unavailable."):
    """Decorator for functions that use email"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            handler = get_degradation_handler()
            return handler.with_fallback(
                "gmail",
                lambda: func(*args, **kwargs),
                fallback_value=fallback_response
            )
        return wrapper
    return decorator


def with_calendar_fallback(fallback_response: str = "Calendar service is temporarily unavailable."):
    """Decorator for functions that use calendar"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            handler = get_degradation_handler()
            return handler.with_fallback(
                "calendar",
                lambda: func(*args, **kwargs),
                fallback_value=fallback_response
            )
        return wrapper
    return decorator


# Example usage in main.py:
"""
from ai.service_degradation import get_degradation_handler, with_ollama_fallback

# In ALICE class:
def __init__(self, ...):
    self.degradation_handler = get_degradation_handler()
    ...

# Use with_fallback for critical operations:
def generate_response(self, user_input: str) -> str:
    return self.degradation_handler.with_fallback(
        "ollama",
        primary_func=lambda: self.llm.generate(user_input),
        fallback_func=lambda: self.conversational_engine.generate_response(user_input),
        fallback_value="I'm having trouble generating a response."
    )

# Or use decorator:
@with_ollama_fallback("I can't access the AI service right now.")
def generate_with_llm(self, prompt: str) -> str:
    return self.llm.generate(prompt)
"""
