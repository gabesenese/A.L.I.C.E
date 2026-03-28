"""Hard contracts between routing, memory, tools, and response generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


@dataclass(frozen=True)
class RouterRequest:
    user_input: str
    context: Dict[str, Any] = field(default_factory=dict)
    turn_number: int = 0


@dataclass(frozen=True)
class RouterDecision:
    route: str
    intent: str
    confidence: float
    needs_clarification: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryRequest:
    query: str
    user_id: str
    max_items: int = 10
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryResult:
    items: List[Dict[str, Any]] = field(default_factory=list)
    source: str = ""
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolInvocation:
    tool_name: str
    action: str
    params: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 30.0
    idempotency_key: Optional[str] = None


@dataclass(frozen=True)
class ToolResult:
    success: bool
    tool_name: str
    action: str
    data: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    confidence: float = 0.0
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResponseRequest:
    user_input: str
    decision: RouterDecision
    memory: MemoryResult
    tool_result: Optional[ToolResult] = None
    style: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResponseOutput:
    text: str
    confidence: float
    requires_follow_up: bool = False
    follow_up_question: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class RoutingBoundary(Protocol):
    def route(self, request: RouterRequest) -> RouterDecision:
        ...


class MemoryBoundary(Protocol):
    def recall(self, request: MemoryRequest) -> MemoryResult:
        ...

    def store(self, item: Dict[str, Any]) -> None:
        ...


class ToolBoundary(Protocol):
    def execute(self, invocation: ToolInvocation) -> ToolResult:
        ...


class ResponseBoundary(Protocol):
    def generate(self, request: ResponseRequest) -> ResponseOutput:
        ...


@dataclass(frozen=True)
class RuntimeBoundaries:
    routing: RoutingBoundary
    memory: MemoryBoundary
    tools: ToolBoundary
    response: ResponseBoundary
