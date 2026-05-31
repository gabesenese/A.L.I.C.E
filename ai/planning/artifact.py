"""First-class artifact tracking for multi-step plan execution.

Every executed step produces an Artifact.  The plan executor collects them
and returns them alongside the normal result dict so callers can inspect
what evidence was gathered, what was verified, and what the final outputs are.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Artifact:
    artifact_id: str = field(default_factory=lambda: f"art-{uuid.uuid4().hex[:8]}")
    step_id: str = ""
    kind: str = "data"      # "data" | "text" | "tool_output" | "verification"
    content: Any = None
    evidence: dict = field(default_factory=dict)
    verified: bool = False
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def verify(self, *, passed: bool, reason: str = "") -> "Artifact":
        """Return a new Artifact with verification result stamped."""
        return Artifact(
            artifact_id=self.artifact_id,
            step_id=self.step_id,
            kind="verification",
            content=self.content,
            evidence={**self.evidence, "verification_reason": reason},
            verified=passed,
            created_at=self.created_at,
        )

    def to_dict(self) -> dict:
        return {
            "artifact_id": self.artifact_id,
            "step_id": self.step_id,
            "kind": self.kind,
            "content": self.content,
            "evidence": self.evidence,
            "verified": self.verified,
            "created_at": self.created_at,
        }
