"""Default adapters for runtime boundary contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any

from .runtime_contracts import (
    MemoryBoundary,
    MemoryRequest,
    MemoryResult,
    ResponseBoundary,
    ResponseOutput,
    ResponseRequest,
    RoutingBoundary,
    RouterDecision,
    RouterRequest,
    ToolBoundary,
    ToolInvocation,
    ToolResult,
)


@dataclass
class CallableRoutingAdapter(RoutingBoundary):
    route_fn: Callable[[RouterRequest], RouterDecision]

    def route(self, request: RouterRequest) -> RouterDecision:
        return self.route_fn(request)


@dataclass
class CallableMemoryAdapter(MemoryBoundary):
    recall_fn: Callable[[MemoryRequest], MemoryResult]
    store_fn: Callable[[Dict[str, Any]], None]

    def recall(self, request: MemoryRequest) -> MemoryResult:
        return self.recall_fn(request)

    def store(self, item: Dict[str, Any]) -> None:
        self.store_fn(item)


@dataclass
class CallableToolAdapter(ToolBoundary):
    execute_fn: Callable[[ToolInvocation], ToolResult]

    def execute(self, invocation: ToolInvocation) -> ToolResult:
        return self.execute_fn(invocation)


@dataclass
class CallableResponseAdapter(ResponseBoundary):
    generate_fn: Callable[[ResponseRequest], ResponseOutput]

    def generate(self, request: ResponseRequest) -> ResponseOutput:
        return self.generate_fn(request)
