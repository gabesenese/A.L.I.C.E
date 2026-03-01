"""
Production-Grade Structured Logging System
JSON logging with context correlation for ELK/Splunk/CloudWatch
"""

import logging
import json
import sys
import traceback
from datetime import datetime
from typing import Dict, Any, Optional
from functools import wraps
import threading
from pathlib import Path


class StructuredLogger:
    """
    Structured logging with automatic context correlation
    Each log entry is JSON with: timestamp, level, message, context, trace_id
    """
    
    def __init__(
        self,
        name: str,
        log_level: str = 'INFO',
        log_file: Optional[str] = None,
        enable_console: bool = True,
        enable_json: bool = True
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self.logger.handlers.clear()  # Remove existing handlers
        
        self.enable_json = enable_json
        self.context = threading.local()  # Thread-local context
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            if enable_json:
                console_handler.setFormatter(JSONFormatter())
            else:
                console_handler.setFormatter(
                    logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
                )
            self.logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            log_path = Path(log_file).parent
            log_path.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(JSONFormatter() if enable_json else
                                     logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
            self.logger.addHandler(file_handler)
    
    def set_context(self, **kwargs):
        """Set context for current thread (trace_id, user_id, session_id, etc)"""
        if not hasattr(self.context, 'data'):
            self.context.data = {}
        self.context.data.update(kwargs)
    
    def clear_context(self):
        """Clear thread-local context"""
        if hasattr(self.context, 'data'):
            self.context.data.clear()
    
    def get_context(self) -> Dict[str, Any]:
        """Get current context"""
        return getattr(self.context, 'data', {}).copy()
    
    def _log(self, level: str, message: str, extra: Optional[Dict] = None, exc_info=None):
        """Internal logging method with context"""
        log_data = {
            'message': message,
            'context': self.get_context(),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if extra:
            log_data.update(extra)
        
        if exc_info:
            log_data['exception'] = {
                'type': exc_info[0].__name__ if exc_info[0] else None,
                'message': str(exc_info[1]) if exc_info[1] else None,
                'traceback': ''.join(traceback.format_exception(*exc_info))
            }
        
        getattr(self.logger, level.lower())(
            json.dumps(log_data) if self.enable_json else message,
            extra={'structured_data': log_data} if not self.enable_json else None
        )
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self._log('DEBUG', message, kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self._log('INFO', message, kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self._log('WARNING', message, kwargs)
    
    def error(self, message: str, exc_info=None, **kwargs):
        """Log error message"""
        self._log('ERROR', message, kwargs, exc_info=exc_info or sys.exc_info())
    
    def critical(self, message: str, exc_info=None, **kwargs):
        """Log critical message"""
        self._log('CRITICAL', message, kwargs, exc_info=exc_info or sys.exc_info())
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback"""
        self.error(message, exc_info=sys.exc_info(), **kwargs)


class JSONFormatter(logging.Formatter):
    """Format log records as JSON"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': self.formatException(record.exc_info)
            }
        
        # Add extra fields
        if hasattr(record, 'structured_data'):
            log_data.update(record.structured_data)
        
        return json.dumps(log_data)


def log_execution(
    logger: Optional[StructuredLogger] = None,
    log_args: bool = False,
    log_result: bool = False
):
    """
    Decorator to log function execution with timing
    
    Usage:
        @log_execution(log_args=True)
        def process_request(user_input: str):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            
            log = logger or get_structured_logger(func.__module__)
            
            log_data = {
                'function': func.__name__,
                'module': func.__module__
            }
            
            if log_args:
                log_data['args'] = str(args)[:200]  # Truncate long args
                log_data['kwargs'] = {k: str(v)[:100] for k, v in kwargs.items()}
            
            try:
                log.debug(f"Executing {func.__name__}", **log_data)
                result = func(*args, **kwargs)
                
                duration = time.time() - start_time
                log_data['duration_ms'] = round(duration * 1000, 2)
                log_data['success'] = True
                
                if log_result and result is not None:
                    log_data['result_type'] = type(result).__name__
                    if isinstance(result, (str, int, float, bool)):
                        log_data['result'] = str(result)[:200]
                
                log.info(f"Completed {func.__name__}", **log_data)
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                log_data['duration_ms'] = round(duration * 1000, 2)
                log_data['success'] = False
                log_data['error'] = str(e)
                
                log.error(f"Failed {func.__name__}", **log_data)
                raise
        
        return wrapper
    return decorator


# Global logger instances
_loggers = {}


def get_structured_logger(
    name: str = 'alice',
    log_level: str = 'INFO',
    log_file: Optional[str] = None
) -> StructuredLogger:
    """Get or create structured logger instance"""
    if name not in _loggers:
        _loggers[name] = StructuredLogger(
            name=name,
            log_level=log_level,
            log_file=log_file or f'logs/{name}.json'
        )
    return _loggers[name]


def configure_logging(
    level: str = 'INFO',
    enable_json: bool = True,
    log_dir: str = 'logs'
):
    """Configure global logging settings"""
    # Create log directory
    Path(log_dir).mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = get_structured_logger(
        name='alice',
        log_level=level,
        log_file=f'{log_dir}/alice.json' if enable_json else f'{log_dir}/alice.log'
    )
    
    # Component-specific loggers
    get_structured_logger('alice.nlp', level, f'{log_dir}/nlp.json')
    get_structured_logger('alice.llm', level, f'{log_dir}/llm.json')
    get_structured_logger('alice.plugins', level, f'{log_dir}/plugins.json')
    get_structured_logger('alice.learning', level, f'{log_dir}/learning.json')
    get_structured_logger('alice.errors', 'WARNING', f'{log_dir}/errors.json')
    
    return root_logger
