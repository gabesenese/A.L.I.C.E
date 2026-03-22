"""Coding-specialized model role for development tasks."""

from __future__ import annotations

import os

from models.base import OllamaRoleModel


class CodingModel(OllamaRoleModel):
    def __init__(self) -> None:
        super().__init__(
            model_name=os.getenv("ALICE_CODING_MODEL", "qwen2.5-coder:7b"),
            system_prompt="You are a senior software engineer assistant. Prefer correct, runnable solutions.",
            reasoning_used=True,
            temperature=0.15,
            timeout_seconds=60,
        )
