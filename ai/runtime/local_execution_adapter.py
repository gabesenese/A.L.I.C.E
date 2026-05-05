from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class LocalExecutionAdapterResult:
    success: bool
    response: str
    operator_context: Dict[str, Any]
    local_execution: Dict[str, Any]
    error: str = ""

