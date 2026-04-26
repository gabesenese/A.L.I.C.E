from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    user_id: str = "anonymous"
    context: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    response: str
    trace_id: str
    requires_follow_up: bool = False
    tools_used: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
