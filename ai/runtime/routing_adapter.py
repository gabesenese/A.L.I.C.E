from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class RoutingAdapterResult:
    route: str
    intent: str
    metadata: Dict[str, Any]

