"""Unified tool execution and normalization layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ai.core.tool_result_verifier import ToolResultVerifier, get_tool_result_verifier


@dataclass
class ToolExecutionOutcome:
    handled: bool
    verified: bool
    result: Dict[str, Any]


class ToolExecutor:
    """Executes plugins through a single normalization + verification path."""

    REQUIRED_PARAMS = {
        "create": ["title"],
        "read": ["target"],
        "delete": ["target"],
    }

    def __init__(self, verifier: Optional[ToolResultVerifier] = None) -> None:
        self.verifier = verifier or get_tool_result_verifier()

    def execute(
        self,
        *,
        plugin_manager: Any,
        intent: str,
        query: str,
        entities: Dict[str, Any],
        context: Dict[str, Any],
    ) -> ToolExecutionOutcome:
        action = self._infer_action(intent)
        missing = self._missing_required_params(action, entities, query)
        if missing:
            result = self._build_result(
                success=False,
                status="failed",
                data={"missing_params": missing},
                message=(
                    f"Missing required parameters for {action}: {', '.join(missing)}. "
                    "Please clarify before I run that tool."
                ),
                confidence=0.2,
                retryable=True,
                side_effects=[],
                plugin=self._infer_plugin(intent),
                action=action,
            )
            verification = self.verifier.verify(result)
            result["verification"] = verification.to_dict()
            return ToolExecutionOutcome(
                handled=False, verified=verification.accepted, result=result
            )

        raw = plugin_manager.execute_for_intent(intent, query, entities, context)
        normalized = self._normalize_result(intent=intent, raw=raw)
        verification = self.verifier.verify(normalized)
        normalized["verification"] = verification.to_dict()

        if not verification.accepted:
            normalized["success"] = False
            normalized["status"] = "verification_failed"
            normalized["retryable"] = True
            normalized["message"] = (
                "I could not verify that tool output was reliable. Please clarify or try again."
            )
            normalized["data"] = {
                **(normalized.get("data") or {}),
                "verification_issues": verification.issues,
                "verification_confidence": verification.confidence,
            }

        handled = raw is not None
        return ToolExecutionOutcome(
            handled=handled, verified=verification.accepted, result=normalized
        )

    def _infer_action(self, intent: str) -> str:
        if ":" in (intent or ""):
            return intent.split(":", 1)[1].strip().lower() or "unknown"
        return "unknown"

    def _infer_plugin(self, intent: str) -> str:
        if ":" in (intent or ""):
            return intent.split(":", 1)[0].strip().lower() or "unknown"
        return "unknown"

    def _missing_required_params(
        self,
        action: str,
        entities: Dict[str, Any],
        query: str,
    ) -> list[str]:
        required = self.REQUIRED_PARAMS.get(action)
        if not required:
            return []

        entities = entities or {}
        query_l = (query or "").lower()
        missing: list[str] = []

        def _has_title() -> bool:
            return bool(
                entities.get("title")
                or entities.get("note_title")
                or entities.get("event")
                or entities.get("subject")
            )

        def _has_target() -> bool:
            return bool(
                entities.get("target")
                or entities.get("note_id")
                or entities.get("filename")
                or entities.get("path")
                or any(
                    tok in query_l
                    for tok in (
                        " it",
                        " this",
                        " that",
                        " note",
                        " file",
                        " email",
                        " event",
                    )
                )
            )

        checks = {
            "title": _has_title,
            "target": _has_target,
        }

        for param in required:
            fn = checks.get(param)
            if fn and not fn():
                missing.append(param)

        return missing

    def _normalize_result(
        self, *, intent: str, raw: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        plugin = self._infer_plugin(intent)
        action = self._infer_action(intent)

        if raw is None:
            return self._build_result(
                success=False,
                status="not_handled",
                data={},
                message="No tool handled this request.",
                confidence=0.1,
                retryable=True,
                side_effects=[],
                plugin=plugin,
                action=action,
            )

        if not isinstance(raw, dict):
            return self._build_result(
                success=False,
                status="failed",
                data={"raw_type": str(type(raw))},
                message="Tool returned an invalid result format.",
                confidence=0.1,
                retryable=True,
                side_effects=[],
                plugin=plugin,
                action=action,
            )

        plugin = str(raw.get("plugin") or plugin)
        action = str(raw.get("action") or action)
        success = bool(raw.get("success", False))
        data = raw.get("data") if isinstance(raw.get("data"), dict) else {}
        message = str(raw.get("message") or raw.get("response") or "")
        confidence = self._coerce_confidence(raw.get("confidence"), success)
        status = self._normalize_status(raw.get("status"), success)
        retryable = bool(raw.get("retryable", not success))
        side_effects = (
            raw.get("side_effects")
            if isinstance(raw.get("side_effects"), list)
            else self._infer_side_effects(action, success)
        )

        normalized = self._build_result(
            success=success,
            status=status,
            data=data,
            message=message,
            confidence=confidence,
            retryable=retryable,
            side_effects=side_effects,
            plugin=plugin,
            action=action,
        )

        # Keep passthrough fields that some downstream components already read.
        if "notes" in raw:
            normalized["notes"] = raw.get("notes")
        if "response" in raw and not normalized.get("message"):
            normalized["message"] = str(raw.get("response") or "")

        return normalized

    def _normalize_status(self, status: Any, success: bool) -> str:
        s = str(status or "").strip().lower()
        if s in {"success", "failed", "partial", "not_handled", "verification_failed"}:
            return s
        return "success" if success else "failed"

    def _coerce_confidence(self, confidence: Any, success: bool) -> float:
        try:
            c = float(confidence)
            return max(0.0, min(1.0, c))
        except Exception:
            return 0.95 if success else 0.35

    def _infer_side_effects(self, action: str, success: bool) -> list[str]:
        if not success:
            return []
        mapping = {
            "create": ["created"],
            "read": ["read"],
            "delete": ["deleted"],
            "update": ["updated"],
            "append": ["updated"],
            "list": ["listed"],
            "search": ["searched"],
        }
        return mapping.get(action, ["tool_executed"])

    def _build_result(
        self,
        *,
        success: bool,
        status: str,
        data: Dict[str, Any],
        message: str,
        confidence: float,
        retryable: bool,
        side_effects: list[str],
        plugin: str,
        action: str,
    ) -> Dict[str, Any]:
        return {
            "success": bool(success),
            "status": str(status),
            "data": data if isinstance(data, dict) else {},
            "message": str(message or ""),
            "response": str(message or ""),
            "confidence": float(confidence),
            "retryable": bool(retryable),
            "side_effects": list(side_effects),
            "plugin": plugin,
            "action": action,
        }


_tool_executor: Optional[ToolExecutor] = None


def get_tool_executor() -> ToolExecutor:
    global _tool_executor
    if _tool_executor is None:
        _tool_executor = ToolExecutor()
    return _tool_executor
