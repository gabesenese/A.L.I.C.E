"""Runtime boundary contracts for thin composition-root architecture."""

from .runtime_contracts import (
    MemoryRequest,
    MemoryResult,
    ResponseOutput,
    ResponseRequest,
    RouterDecision,
    RouterRequest,
    RuntimeBoundaries,
    ToolInvocation,
    ToolResult,
)
from .default_adapters import (
    CallableMemoryAdapter,
    CallableResponseAdapter,
    CallableRoutingAdapter,
    CallableToolAdapter,
)

__all__ = [
    "RouterRequest",
    "RouterDecision",
    "MemoryRequest",
    "MemoryResult",
    "ToolInvocation",
    "ToolResult",
    "ResponseRequest",
    "ResponseOutput",
    "RuntimeBoundaries",
    "CallableRoutingAdapter",
    "CallableMemoryAdapter",
    "CallableToolAdapter",
    "CallableResponseAdapter",
]
