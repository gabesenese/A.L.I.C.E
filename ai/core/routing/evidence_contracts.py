from __future__ import annotations

import re
from typing import Any, Dict


class EvidenceContracts:
    FILE_LIST_PATTERN = re.compile(
        r"\b(?:what files|show files|show me files|list files|which files|what files does alice have|what files can you inspect)\b",
        re.IGNORECASE,
    )
    FILE_TARGET_PATTERN = re.compile(r"\b[a-z0-9_./\\-]+\.[a-z0-9]{1,8}\b", re.IGNORECASE)

    @classmethod
    def has_explicit_file_target(cls, text: str) -> bool:
        return bool(cls.FILE_TARGET_PATTERN.search(str(text or "")))

    @classmethod
    def evaluate(cls, *, intent: str, user_input: str, active_mode: str = "") -> Dict[str, Any]:
        normalized = str(intent or "").strip().lower()
        text = str(user_input or "")
        result = {
            "intent": normalized,
            "accepted": True,
            "reason": "ok",
            "reroute_intent": "",
            "file_tool_vetoed": False,
        }

        if normalized == "file_operations:read":
            if not cls.has_explicit_file_target(text):
                reroute = "code:list_files" if cls.FILE_LIST_PATTERN.search(text) else "code:request"
                result.update(
                    {
                        "accepted": False,
                        "reason": "no_explicit_file_target",
                        "reroute_intent": reroute,
                        "file_tool_vetoed": True,
                    }
                )

        if normalized == "code:analyze_file" and not cls.has_explicit_file_target(text):
            if active_mode == "code_inspection":
                result["accepted"] = True
            else:
                result.update(
                    {
                        "accepted": False,
                        "reason": "code_analyze_requires_target",
                        "reroute_intent": "code:request",
                    }
                )

        if normalized == "code:list_files" and not cls.FILE_LIST_PATTERN.search(text):
            result.update(
                {
                    "accepted": False,
                    "reason": "code_list_files_requires_workspace_question",
                    "reroute_intent": "code:request",
                }
            )

        return result
