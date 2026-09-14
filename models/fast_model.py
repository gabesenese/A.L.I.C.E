"""Fast/cheap model role for simple daily requests."""

from __future__ import annotations

import os

from ai.core import persona
from models.base import OllamaRoleModel


class FastModel(OllamaRoleModel):
    def __init__(self) -> None:
        super().__init__(
            model_name=os.getenv("ALICE_FAST_MODEL", "llama3.2:3b"),
            system_prompt=persona.for_brief_reply(),
            reasoning_used=False,
            temperature=0.2,
            timeout_seconds=20,
        )
