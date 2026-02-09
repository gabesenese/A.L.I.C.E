"""
Structured Error Model for A.L.I.C.E
Provides typed exceptions for different subsystem failures
"""

from enum import Enum
from typing import Optional, Dict, Any


class ErrorSeverity(Enum):
    """Severity level for errors"""
    INFO = "info"           # Non-critical, informational
    WARNING = "warning"     # Degraded functionality but system continues
    ERROR = "error"         # Significant failure, feature unavailable
    CRITICAL = "critical"   # System-level failure


class ALICEError(Exception):
    """Base exception for all ALICE errors"""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        user_message: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = True
    ):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.user_message = user_message or self._default_user_message()
        self.context = context or {}
        self.recoverable = recoverable
    
    def _default_user_message(self) -> str:
        """Default user-friendly message"""
        return "I encountered an issue processing that request."
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "severity": self.severity.value,
            "user_message": self.user_message,
            "context": self.context,
            "recoverable": self.recoverable
        }


class NLPError(ALICEError):
    """Errors from NLP processing pipeline"""
    
    def _default_user_message(self) -> str:
        return "I had trouble understanding that request. Could you rephrase it?"


class IntentDetectionError(NLPError):
    """Failed to detect user intent"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.WARNING,
            **kwargs
        )


class EntityExtractionError(NLPError):
    """Failed to extract required entities"""
    
    def __init__(self, message: str, missing_entities: list = None, **kwargs):
        context = kwargs.pop('context', {})
        context['missing_entities'] = missing_entities or []
        super().__init__(
            message,
            severity=ErrorSeverity.WARNING,
            context=context,
            **kwargs
        )
    
    def _default_user_message(self) -> str:
        missing = ', '.join(self.context.get('missing_entities', []))
        if missing:
            return f"I need more information: {missing}"
        return "I need more details to complete that request."


class ToolError(ALICEError):
    """Errors from plugin/tool execution"""
    
    def __init__(self, tool_name: str, message: str, **kwargs):
        context = kwargs.pop('context', {})
        context['tool_name'] = tool_name
        super().__init__(message, context=context, **kwargs)
    
    def _default_user_message(self) -> str:
        tool_name = self.context.get('tool_name', 'that feature')
        return f"I couldn't complete the {tool_name} operation right now."


class EmailError(ToolError):
    """Email plugin failures"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__('email', message, **kwargs)
    
    def _default_user_message(self) -> str:
        return "I couldn't access your email right now. Please check your connection and credentials."


class CalendarError(ToolError):
    """Calendar plugin failures"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__('calendar', message, **kwargs)
    
    def _default_user_message(self) -> str:
        return "I couldn't access your calendar. Please check your connection and credentials."


class FileOperationError(ToolError):
    """File operations failures"""
    
    def __init__(self, message: str, file_path: Optional[str] = None, **kwargs):
        context = kwargs.pop('context', {})
        if file_path:
            context['file_path'] = file_path
        super().__init__('file_operations', message, context=context, **kwargs)
    
    def _default_user_message(self) -> str:
        file_path = self.context.get('file_path')
        if file_path:
            return f"I couldn't access the file: {file_path}"
        return "I encountered a file operation error."


class NetworkError(ToolError):
    """Network-related tool failures"""
    
    def __init__(self, tool_name: str, message: str, **kwargs):
        super().__init__(tool_name, message, severity=ErrorSeverity.WARNING, **kwargs)
    
    def _default_user_message(self) -> str:
        return "I'm having trouble connecting to that service. Please check your network connection."


class LLMError(ALICEError):
    """Errors from LLM engine"""
    
    def _default_user_message(self) -> str:
        return "I'm having trouble generating a response. The AI service may be unavailable."


class OllamaConnectionError(LLMError):
    """Cannot connect to Ollama service"""
    
    def __init__(self, message: str = "Cannot connect to Ollama", **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.ERROR,
            recoverable=True,
            **kwargs
        )
    
    def _default_user_message(self) -> str:
        return "I can't reach the AI service. Make sure Ollama is running."


class LLMGenerationError(LLMError):
    """LLM failed to generate response"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.WARNING,
            **kwargs
        )


class MemoryError(ALICEError):
    """Errors from memory system"""
    
    def _default_user_message(self) -> str:
        return "I had trouble accessing my memory system."


class MemoryStorageError(MemoryError):
    """Failed to store memory"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity=ErrorSeverity.WARNING, **kwargs)


class MemoryRetrievalError(MemoryError):
    """Failed to retrieve memory"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity=ErrorSeverity.WARNING, **kwargs)


class ContextError(ALICEError):
    """Errors from context management"""
    
    def _default_user_message(self) -> str:
        return "I lost track of our conversation context."


class RoutingError(ALICEError):
    """Errors in request routing logic"""
    
    def _default_user_message(self) -> str:
        return "I'm not sure how to handle that request."


class ConfigurationError(ALICEError):
    """System configuration errors"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.CRITICAL,
            recoverable=False,
            **kwargs
        )
    
    def _default_user_message(self) -> str:
        return "There's a configuration problem. Please check the system setup."


class IOError(ALICEError):
    """File I/O errors"""
    
    def __init__(self, message: str, file_path: Optional[str] = None, **kwargs):
        context = kwargs.pop('context', {})
        if file_path:
            context['file_path'] = file_path
        super().__init__(message, context=context, **kwargs)
    
    def _default_user_message(self) -> str:
        file_path = self.context.get('file_path')
        if file_path:
            return f"I couldn't read or write to: {file_path}"
        return "I encountered a file system error."


# Centralized error handler
class ErrorHandler:
    """Centralized error handling and user message generation"""
    
    @staticmethod
    def handle_error(error: Exception, logger=None) -> str:
        """
        Handle any exception and return user-friendly message
        
        Args:
            error: The exception to handle
            logger: Optional logger for recording error details
            
        Returns:
            User-friendly error message
        """
        if isinstance(error, ALICEError):
            # Log structured error
            if logger:
                logger.error(
                    f"{error.__class__.__name__}: {error.message}",
                    extra=error.to_dict()
                )
            return error.user_message
        else:
            # Unknown error - log and return generic message
            if logger:
                logger.exception(f"Unexpected error: {str(error)}")
            return "I encountered an unexpected issue. Please try again."
    
    @staticmethod
    def is_recoverable(error: Exception) -> bool:
        """Check if error is recoverable"""
        if isinstance(error, ALICEError):
            return error.recoverable
        return True  # Assume unknown errors are recoverable
    
    @staticmethod
    def get_severity(error: Exception) -> ErrorSeverity:
        """Get error severity"""
        if isinstance(error, ALICEError):
            return error.severity
        return ErrorSeverity.ERROR
