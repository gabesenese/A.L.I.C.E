from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class ResponseAdapterResult:
    text: str
    metadata: Dict[str, Any]
    requires_follow_up: bool = False

