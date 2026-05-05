from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class ToolExecutionAdapterResult:
    success: bool
    payload: Dict[str, Any]
    error: str = ""

