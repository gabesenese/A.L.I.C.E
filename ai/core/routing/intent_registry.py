from __future__ import annotations


class IntentRegistry:
    CORE_INTENTS = {
        "conversation:general",
        "conversation:project_work_session",
        "conversation:collaborative_reasoning",
        "conversation:understanding_review",
        "conversation:memory_recall",
        "code:request",
        "code:list_files",
        "code:read_file",
        "code:analyze_file",
        "weather:current",
        "weather:forecast",
        "notes:create",
        "notes:read",
        "file_operations:read",
    }

    @classmethod
    def supports(cls, intent: str) -> bool:
        return str(intent or "") in cls.CORE_INTENTS
