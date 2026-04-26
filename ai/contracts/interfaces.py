from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class RouteDecision(Enum):
    TOOL = "tool"
    PLUGIN = "plugin"
    CONVERSATION = "conversation"
    CLARIFY = "clarify"
    REFUSE = "refuse"


@dataclass(frozen=True)
class RouteResult:
    intent: str
    confidence: float
    route: RouteDecision
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChatResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SlotFillerResult:
    slots: dict[str, str]
    confidence: float


@dataclass(frozen=True)
class TemporalParseResult:
    normalized_text: str
    iso_datetime: str | None
    confidence: float


@dataclass(frozen=True)
class EmotionResult:
    labels: list[str]
    sentiment: dict[str, float]


class SafetyLevel(Enum):
    CHAT = 0
    SAFE_TOOLS = 1
    FULL_CONTROL = 2


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    level: SafetyLevel
    reason: str


@dataclass(frozen=True)
class VerificationResult:
    accepted: bool
    reason: str
    confidence: float


@dataclass(frozen=True)
class ExecutionResult:
    success: bool
    output: str
    tool_calls: list[ToolCall]
    latency_ms: float
    requires_follow_up: bool


@dataclass(frozen=True)
class TurnStateMachineResult:
    phase: str
    verification: VerificationResult
    execution: ExecutionResult


class ExecutiveControllerProtocol(Protocol):
    def build_state(
        self,
        user_input: str,
        intent: str,
        action_discipline: dict[str, Any],
    ) -> Any: ...


class CompanionRuntimeProtocol(Protocol):
    def start_turn(self, user_input: str, user_id: str) -> Any: ...


class NLPEngineProtocol(Protocol):
    async def process(self, text: str, history: list[str]) -> RouteResult: ...


class LLMEngineProtocol(Protocol):
    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
    ) -> ChatResponse: ...

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any: ...


class MemoryStoreProtocol(Protocol):
    async def store(
        self,
        trace_id: str,
        text: str,
        metadata: dict[str, Any],
    ) -> None: ...

    async def query(
        self,
        text: str,
        n_results: int = 5,
    ) -> list[dict[str, Any]]: ...
