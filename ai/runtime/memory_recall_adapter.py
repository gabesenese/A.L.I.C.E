from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class MemoryRecallAdapterResult:
    items: List[Dict[str, Any]]
    confidence: float
    metadata: Dict[str, Any]

