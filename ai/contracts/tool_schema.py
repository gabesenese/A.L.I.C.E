"""Schema validation helpers for tool invocation and tool results."""

from __future__ import annotations

from typing import Any, Dict


class ToolSchemaValidationError(ValueError):
    """Raised when tool schema does not satisfy required contract."""


def validate_tool_invocation_payload(payload: Dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ToolSchemaValidationError("tool invocation payload must be a dict")
    if not str(payload.get("tool_name") or "").strip():
        raise ToolSchemaValidationError("tool_name is required")
    if not str(payload.get("action") or "").strip():
        raise ToolSchemaValidationError("action is required")
    params = payload.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ToolSchemaValidationError("params must be a dict")


def validate_tool_result_payload(payload: Dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ToolSchemaValidationError("tool result payload must be a dict")
    if "success" not in payload:
        raise ToolSchemaValidationError("success field is required")
    if not isinstance(payload.get("success"), bool):
        raise ToolSchemaValidationError("success must be bool")
    if not str(payload.get("tool_name") or "").strip():
        raise ToolSchemaValidationError("tool_name is required")
    if not str(payload.get("action") or "").strip():
        raise ToolSchemaValidationError("action is required")
    data = payload.get("data", {})
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ToolSchemaValidationError("data must be a dict")
    diagnostics = payload.get("diagnostics", {})
    if diagnostics is None:
        diagnostics = {}
    if not isinstance(diagnostics, dict):
        raise ToolSchemaValidationError("diagnostics must be a dict")
