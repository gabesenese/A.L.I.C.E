from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List


class CodeAnalyzer:
    def responsibility(self, rel: str, text: str) -> str:
        low = (rel + "\n" + text[:2000]).lower()
        if "route" in low or "intent" in low:
            return "routing"
        if "runtime" in low or "orchestrator" in low or "pipeline" in low:
            return "runtime pipeline"
        if "memory" in low:
            return "memory"
        if "plugin" in low or "tool" in low:
            return "plugin/tool"
        if Path(rel).name in {"main.py", "alice.py", "legacy_main.py"}:
            return "app entrypoint"
        if "/test" in rel.lower() or rel.lower().startswith("tests/"):
            return "tests"
        if "legacy" in rel.lower() or "archive" in rel.lower():
            return "legacy/dead code candidate"
        return "general module"

    def stats(self, text: str) -> Dict[str, Any]:
        lines = text.splitlines()
        fallback_markers = [
            "please rephrase",
            "share the exact outcome",
            "could not verify",
            "clarify what you want",
            "ask me to",
        ]
        hardcoded_markers = [
            "I can inspect",
            "I could not verify",
            "Share the exact outcome you want",
        ]
        lower = text.lower()
        return {
            "line_count": len(lines),
            "char_count": len(text),
            "import_count": len(re.findall(r"^\s*(?:from\s+\S+\s+import|import\s+\S+)", text, flags=re.MULTILINE)),
            "class_count": len(re.findall(r"^\s*class\s+[A-Za-z_]\w*", text, flags=re.MULTILINE)),
            "function_count": len(re.findall(r"^\s*def\s+[A-Za-z_]\w*\s*\(", text, flags=re.MULTILINE)),
            "todo_count": len(re.findall(r"\b(?:TODO|FIXME)\b", text)),
            "fallback_phrase_count": sum(lower.count(marker) for marker in fallback_markers),
            "hardcoded_response_phrase_count": sum(text.count(marker) for marker in hardcoded_markers),
            "regex_count": len(re.findall(r"re\.(?:search|match|findall|compile)\s*\(", text)),
            "route_intent_mentions": len(re.findall(r"\b(?:route|intent|local|tool|clarify)\b", lower)),
            "broad_keyword_matching_risk": bool(
                re.search(r"\bif\s+['\"].+['\"]\s+in\s+[a-zA-Z_][a-zA-Z0-9_]*", text)
                or " in text.lower()" in lower
            ),
            "large_file_warning": len(lines) > 800 or len(text) > 60000,
        }

    def risk_flags(self, text: str, rel: str, stats: Dict[str, Any]) -> List[str]:
        flags: List[str] = []
        low = text.lower()
        if stats.get("fallback_phrase_count", 0) > 0:
            flags.append("contains fallback-phrase response logic")
        if stats.get("broad_keyword_matching_risk"):
            flags.append("contains broad substring matching that can over-trigger routes")
        if stats.get("function_count", 0) >= 25:
            flags.append("god-file risk: many function responsibilities")
        if "route=" in low and "intent=" in low:
            flags.append("contains direct hardcoded route/intent decisions")
        if "ask me to" in low:
            flags.append("passive capability phrasing detected")
        if "local" in low and "context" in low and "memory_count" in low:
            flags.append("local route may rely on shallow execution context")
        if "legacy" in rel.lower():
            flags.append("legacy file candidate for deprecation review")
        return flags[:6]

    def suggest_next_files(self, rel: str, text: str, files: List[str]) -> List[str]:
        picks: List[str] = []
        import_lines = re.findall(r"^\s*from\s+([a-zA-Z0-9_\.]+)\s+import|^\s*import\s+([a-zA-Z0-9_\.]+)", text, flags=re.MULTILINE)
        candidates = []
        for left, right in import_lines:
            token = left or right
            if token:
                candidates.append(token.replace(".", "/") + ".py")
        focus = [
            "ai/runtime/turn_orchestrator.py",
            "ai/runtime/contract_pipeline.py",
            "ai/runtime/alice_contract_factory.py",
            "ai/runtime/companion_runtime.py",
            "ai/core/route_coordinator.py",
            "ai/memory/memory_extractor.py",
            "ai/memory/personal_memory.py",
        ]
        for c in candidates + focus:
            for f in files:
                if f.lower().endswith(c.lower()) and f != rel and f not in picks:
                    picks.append(f)
                    break
            if len(picks) >= 5:
                break
        return picks

