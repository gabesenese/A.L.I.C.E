from __future__ import annotations


class IntentRegistry:
    INTENT_DEFS = {
        "conversation:general": {"route": "llm", "requires_target": False, "safe_without_target": True, "tool_or_local": "llm"},
        "conversation:project_work_session": {"route": "llm", "requires_target": False, "safe_without_target": True, "tool_or_local": "llm"},
        "conversation:collaborative_reasoning": {"route": "llm", "requires_target": False, "safe_without_target": True, "tool_or_local": "llm"},
        "conversation:understanding_review": {"route": "llm", "requires_target": False, "safe_without_target": True, "tool_or_local": "llm"},
        "conversation:memory_recall": {"route": "llm", "requires_target": False, "safe_without_target": True, "tool_or_local": "llm"},
        "code:request": {"route": "local", "requires_target": False, "safe_without_target": True, "tool_or_local": "local"},
        "code:list_files": {"route": "local", "requires_target": False, "safe_without_target": True, "tool_or_local": "local"},
        "code:read_file": {"route": "local", "requires_target": True, "safe_without_target": False, "tool_or_local": "local"},
        "code:analyze_file": {"route": "local", "requires_target": True, "safe_without_target": False, "tool_or_local": "local"},
        "weather:current": {"route": "tool", "requires_target": False, "safe_without_target": False, "tool_or_local": "tool"},
        "weather:forecast": {"route": "tool", "requires_target": False, "safe_without_target": False, "tool_or_local": "tool"},
        "notes:create": {"route": "tool", "requires_target": False, "safe_without_target": False, "tool_or_local": "tool"},
        "notes:read": {"route": "tool", "requires_target": False, "safe_without_target": False, "tool_or_local": "tool"},
        "file_operations:read": {"route": "tool", "requires_target": True, "safe_without_target": False, "tool_or_local": "tool"},
    }
    CORE_INTENTS = set(INTENT_DEFS.keys())

    @classmethod
    def supports(cls, intent: str) -> bool:
        return str(intent or "") in cls.CORE_INTENTS

    @classmethod
    def get(cls, intent: str):
        return dict(cls.INTENT_DEFS.get(str(intent or ""), {}))
