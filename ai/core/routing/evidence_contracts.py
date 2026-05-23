from __future__ import annotations

import re
from typing import Any, Dict


class EvidenceContracts:
    FILE_LIST_PATTERN = re.compile(
        r"\b(?:what files|show files|show me files|list files|which files|what files does alice have|what files can you inspect)\b",
        re.IGNORECASE,
    )
    FILE_TARGET_PATTERN = re.compile(
        r"\b[a-z0-9_./\\-]+\.[a-z0-9]{1,8}\b", re.IGNORECASE
    )
    NOTES_SIGNAL = re.compile(r"\b(note|notes|notepad)\b", re.IGNORECASE)
    WEATHER_SIGNAL = re.compile(
        r"\b(weather|forecast|rain|snow|temperature|wind|humidity|coat|umbrella)\b",
        re.IGNORECASE,
    )
    MEMORY_SIGNAL = re.compile(
        r"\b(remember|recall|what did i|what do you remember|my personal life)\b",
        re.IGNORECASE,
    )
    CONTINUATION_SIGNAL = re.compile(
        r"\b(where were we|what'?s next|continue|pick up where we left off|keep going)\b",
        re.IGNORECASE,
    )

    @classmethod
    def has_explicit_file_target(cls, text: str) -> bool:
        return bool(cls.FILE_TARGET_PATTERN.search(str(text or "")))

    @classmethod
    def evaluate(
        cls,
        *,
        intent: str,
        user_input: str,
        active_mode: str = "",
        operator_state: Dict[str, Any] | None = None,
        project_memory: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        normalized = str(intent or "").strip().lower()
        text = str(user_input or "")
        low = text.lower()
        state = dict(operator_state or {})
        proj = dict(project_memory or {})
        active_objective = str(
            state.get("active_objective") or proj.get("active_objective") or ""
        ).strip()

        result = {
            "intent": normalized,
            "accepted": True,
            "reason": "ok",
            "reroute_intent": "",
            "evidence_score": 0.6,
            "missing_slots": [],
            "requires_approval": False,
            "file_tool_vetoed": False,
            "unsafe_action_vetoed": False,
            "clarification_allowed": False,
        }

        destructive = any(
            token in low
            for token in (
                "delete all",
                "wipe",
                "format disk",
                "drop database",
                "bypass security",
            )
        )
        if destructive:
            result.update(
                {
                    "accepted": False,
                    "reason": "unsafe_action_vetoed",
                    "unsafe_action_vetoed": True,
                    "requires_approval": True,
                    "evidence_score": 0.0,
                }
            )
            return result

        if normalized.startswith("file_operations:"):
            action = normalized.split(":", 1)[1]
            if action in {"write", "delete"}:
                result["requires_approval"] = True
            if action != "list" and not cls.has_explicit_file_target(text):
                reroute = (
                    "code:list_files"
                    if cls.FILE_LIST_PATTERN.search(text)
                    else "code:request"
                )
                result.update(
                    {
                        "accepted": False,
                        "reason": "no_explicit_file_target",
                        "reroute_intent": reroute,
                        "file_tool_vetoed": True,
                        "missing_slots": ["file_target"],
                        "clarification_allowed": True,
                        "evidence_score": 0.25,
                    }
                )
            return result

        if normalized in {"code:request", "code:list_files"}:
            result["evidence_score"] = 0.72 if "file" in low or "code" in low else 0.55
            return result

        if normalized in {"code:read_file", "code:analyze_file"}:
            if not cls.has_explicit_file_target(text):
                has_context_target = bool(state.get("last_inspected_file"))
                if not has_context_target:
                    result.update(
                        {
                            "accepted": False,
                            "reason": "code_target_required",
                            "reroute_intent": "code:list_files",
                            "missing_slots": ["file_target"],
                            "clarification_allowed": True,
                            "evidence_score": 0.3,
                        }
                    )
            result["evidence_score"] = 0.75
            return result

        if normalized == "memory:recall":
            if not cls.MEMORY_SIGNAL.search(text):
                result.update(
                    {
                        "accepted": False,
                        "reason": "memory_question_evidence_missing",
                        "clarification_allowed": True,
                        "evidence_score": 0.2,
                    }
                )
            else:
                result["evidence_score"] = 0.8
            return result

        if normalized.startswith("notes:"):
            if not cls.NOTES_SIGNAL.search(text):
                result.update(
                    {
                        "accepted": False,
                        "reason": "notes_domain_evidence_missing",
                        "clarification_allowed": True,
                        "evidence_score": 0.25,
                    }
                )
            else:
                result["evidence_score"] = 0.78
            return result

        if normalized.startswith("weather:"):
            if not cls.WEATHER_SIGNAL.search(text):
                result.update(
                    {
                        "accepted": False,
                        "reason": "weather_evidence_missing",
                        "clarification_allowed": True,
                        "evidence_score": 0.25,
                    }
                )
            else:
                result["evidence_score"] = 0.82
            return result

        if normalized in {
            "operator:next_step",
            "operator:project_status",
            "operator:continue",
            "operator:execute_recommended_action",
            "operator:explain_recommendation",
            "self_improvement:audit",
            "self_improvement:status",
            "self_improvement:brief",
        }:
            has_state = bool(
                active_objective
                or state.get("current_focus")
                or proj.get("current_focus")
            )
            if not has_state:
                result.update(
                    {
                        "accepted": False,
                        "reason": "operator_state_missing",
                        "reroute_intent": "conversation:goal_statement",
                        "clarification_allowed": True,
                        "evidence_score": 0.3,
                    }
                )
            else:
                result["evidence_score"] = 0.86
            if normalized == "operator:continue" and not cls.CONTINUATION_SIGNAL.search(
                text
            ):
                result["evidence_score"] = 0.72
            if normalized.startswith("self_improvement:"):
                result["evidence_score"] = 0.84
            return result

        if normalized == "conversation:greeting":
            # Guardrail metadata for consumers: greeting path must not use broad vector memory.
            result["evidence_score"] = 0.9
            return result

        return result
