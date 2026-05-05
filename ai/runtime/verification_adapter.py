from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class VerificationAdapterResult:
    accepted: bool
    reason: str
    confidence: float
    diagnostics: Dict[str, Any]

