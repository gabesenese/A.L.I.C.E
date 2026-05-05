from __future__ import annotations

from typing import Any, Dict, List


class CodeResponseBuilder:
    def build_analysis_response(
        self,
        *,
        rel: str,
        responsibility: str,
        stats: Dict[str, Any],
        risks: List[str],
        suggested: List[str],
    ) -> str:
        parts = [
            f"Inspected `{rel}`.",
            f"Primary responsibility: {responsibility}.",
            (
                "Structural stats: "
                f"{stats['line_count']} lines, {stats['char_count']} chars, "
                f"{stats['import_count']} imports, {stats['class_count']} classes, "
                f"{stats['function_count']} functions, {stats['todo_count']} TODO/FIXME, "
                f"{stats['fallback_phrase_count']} fallback phrases."
            ),
        ]
        if risks:
            parts.append("Top risks: " + "; ".join(risks[:3]) + ".")
        if suggested:
            parts.append("Recommended next files:\n- " + "\n- ".join(suggested[:5]))
        if stats.get("large_file_warning"):
            parts.append("Warning: large file; split review into targeted sections.")
        return " ".join(parts)

