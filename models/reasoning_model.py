"""Reasoning model role for complex or multi-step requests."""

from __future__ import annotations

import os

from models.base import OllamaRoleModel


class ReasoningModel(OllamaRoleModel):
    def __init__(self) -> None:
        super().__init__(
            model_name=os.getenv("ALICE_REASONING_MODEL", "llama3.1:8b"),
            system_prompt="You are a careful reasoning assistant. Be accurate and structured.",
            reasoning_used=True,
            temperature=0.25,
            timeout_seconds=45,
        )
