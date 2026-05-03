"""Local runtime actions for operator-style code inspection."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List


class LocalActionExecutor:
    def __init__(self, alice: Any):
        self.alice = alice
        self.root = Path.cwd()

    def _list_files(self, limit: int = 400) -> List[str]:
        refl = getattr(self.alice, "self_reflection", None)
        if refl and hasattr(refl, "list_codebase"):
            try:
                rows = list(refl.list_codebase() or [])
                out = []
                for row in rows[:limit]:
                    path = row.get("path") if isinstance(row, dict) else str(row)
                    if path:
                        out.append(str(path).replace("\\", "/"))
                if out:
                    return out
            except Exception:
                pass
        out: List[str] = []
        for dirpath, _, filenames in os.walk(self.root):
            if any(skip in dirpath.lower() for skip in (".git", ".venv", "__pycache__")):
                continue
            for name in filenames:
                if name.endswith((".py", ".md", ".json", ".yaml", ".yml")):
                    rel = str(Path(dirpath, name).relative_to(self.root)).replace("\\", "/")
                    out.append(rel)
                    if len(out) >= limit:
                        return out
        return out

    def _resolve_target(self, target: str, files: List[str]) -> Dict[str, Any]:
        target = str(target or "").strip().replace("\\", "/")
        exact = [f for f in files if f.lower() == target.lower() or f.lower().endswith("/" + target.lower())]
        if exact:
            return {"file_exists": True, "resolved": exact[0], "close_matches": []}
        basename = target.split("/")[-1].lower()
        close = [f for f in files if basename and basename in f.lower()][:8]
        return {"file_exists": False, "resolved": "", "close_matches": close}

    def _read_file_text(self, rel_path: str, max_chars: int = 12000) -> str:
        p = self.root / rel_path
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(max_chars)

    def _extract_target_from_query(self, query: str) -> str:
        m = re.search(r"([a-zA-Z0-9_./\\-]+\.py)\b", str(query or ""))
        if m:
            return m.group(1)
        return ""

    def execute(self, *, action: str, query: str, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        ctx = dict(context or {})
        files = self._list_files()
        operator_context = {
            "active_capability": "code_inspection",
            "awaiting_target": action in {"code:request", "code:list_files"},
            "continuation_from_previous_turn": bool(ctx.get("continuation_from_previous_turn")),
            "inferred_target_file": "",
            "file_exists": False,
            "close_matches": [],
        }
        local_execution = {
            "action": action,
            "file_count": len(files),
            "inspected_file": "",
            "success": False,
            "error": "",
        }

        if action == "code:list_files":
            preview = files[: min(15, len(files))]
            text = "I can inspect these workspace files:\n- " + "\n- ".join(preview) if preview else "I do not currently see workspace files to inspect."
            local_execution["success"] = bool(preview)
            return {"success": True, "response": text, "operator_context": operator_context, "local_execution": local_execution}

        if action == "code:request":
            focus = [
                f for f in files
                if any(k in f.lower() for k in ("contract_pipeline.py", "turn_orchestrator.py", "alice_contract_factory.py", "route_coordinator.py", "companion_runtime.py"))
            ]
            focus = focus[:5] if focus else files[:5]
            if focus:
                text = (
                    "I can inspect local source code in this workspace and inspect the local workspace. "
                    "Good next targets are:\n- " + "\n- ".join(focus) + "\nTell me one file to inspect, or ask me to list more files."
                )
            else:
                text = "I can attempt local inspection, but I do not currently see workspace files."
            local_execution["success"] = bool(focus)
            operator_context["awaiting_target"] = True
            return {"success": True, "response": text, "operator_context": operator_context, "local_execution": local_execution}

        if action in {"code:analyze_file", "code:read_file"}:
            target = self._extract_target_from_query(query) or str(ctx.get("target_file") or "")
            operator_context["inferred_target_file"] = target
            resolved = self._resolve_target(target, files) if target else {"file_exists": False, "resolved": "", "close_matches": files[:8]}
            operator_context["file_exists"] = bool(resolved["file_exists"])
            operator_context["close_matches"] = list(resolved["close_matches"])
            if not resolved["file_exists"]:
                local_execution["error"] = "target_not_found"
                return {
                    "success": False,
                    "response": "",
                    "error": f"File not found: {target}" if target else "No file target provided.",
                    "operator_context": operator_context,
                    "local_execution": local_execution,
                }
            rel = str(resolved["resolved"])
            operator_context["awaiting_target"] = False
            local_execution["inspected_file"] = rel
            try:
                text = self._read_file_text(rel)
            except Exception as exc:
                local_execution["error"] = str(exc)
                return {"success": False, "response": "", "error": f"Could not read {rel}: {exc}", "operator_context": operator_context, "local_execution": local_execution}
            lines = text.splitlines()
            summary = f"Analyzed {rel}: {len(lines)} lines. "
            if lines:
                summary += "Key observations: file is readable and ready for deeper inspection."
            local_execution["success"] = True
            return {"success": True, "response": summary, "operator_context": operator_context, "local_execution": local_execution}

        if action == "system:location" and hasattr(self.alice, "_build_location_payload"):
            payload = self.alice._build_location_payload()
            local_execution["success"] = bool(payload)
            return {"success": True, "response": "", "data": payload, "operator_context": operator_context, "local_execution": local_execution}

        if action == "freshness:current_events":
            local_execution["success"] = True
            return {"success": True, "response": "", "operator_context": operator_context, "local_execution": local_execution}

        return {"success": False, "response": "", "error": f"Unsupported local action: {action}", "operator_context": operator_context, "local_execution": local_execution}
