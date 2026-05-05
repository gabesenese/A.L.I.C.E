from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict

from ai.runtime.local_actions.code_analyzer import CodeAnalyzer
from ai.runtime.local_actions.code_response_builder import CodeResponseBuilder
from ai.runtime.local_actions.file_index import FileIndex
from ai.runtime.local_actions.file_resolver import FileResolver


class LocalActionExecutor:
    def __init__(self, alice: Any):
        self.alice = alice
        self.root = self._resolve_root()
        self.file_index = FileIndex(alice, self.root, ttl_seconds=45)
        self.file_resolver = FileResolver()
        self.code_analyzer = CodeAnalyzer()
        self.response_builder = CodeResponseBuilder()

    def invalidate_file_cache(self) -> None:
        self.file_index.invalidate()

    def _resolve_root(self) -> Path:
        refl = getattr(self.alice, "self_reflection", None)
        base_path = getattr(refl, "base_path", None) if refl is not None else None
        if base_path:
            try:
                return Path(str(base_path)).resolve()
            except Exception:
                pass
        project_root = getattr(self.alice, "PROJECT_ROOT", None)
        if project_root:
            try:
                return Path(str(project_root)).resolve()
            except Exception:
                pass
        return Path.cwd()

    def _read_file_text(self, rel_path: str, max_chars: int = 12000) -> str:
        p = self.root / rel_path
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(max_chars)

    def _extract_target_from_query(self, query: str) -> str:
        m = re.search(r"([a-zA-Z0-9_./\\-]+\.[a-zA-Z0-9]{1,8})\b", str(query or ""))
        if m:
            return m.group(1)
        return ""

    def execute(self, *, action: str, query: str, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        ctx = dict(context or {})
        files = self.file_index.list_files()
        decision_meta = dict(ctx.get("decision_metadata") or {})
        operator_meta = dict(ctx.get("operator_context") or {})
        mode = str(operator_meta.get("active_mode") or "code_inspection")
        operator_context = {
            "active_capability": "code_inspection",
            "active_mode": mode,
            "awaiting_target": action in {"code:request", "code:list_files"},
            "continuation_from_previous_turn": bool(
                ctx.get("continuation_from_previous_turn")
                or operator_meta.get("continuation_from_previous_turn")
            ),
            "inferred_target_file": "",
            "file_exists": False,
            "close_matches": [],
            "last_route": str(ctx.get("route") or ""),
            "last_intent": str(ctx.get("intent") or ""),
        }
        local_execution = {
            "action": action,
            "workspace_file_count": len(files),
            "inspected_file": "",
            "success": False,
            "error": "",
        }

        if action == "code:list_files":
            preview = files[: min(15, len(files))]
            focus_candidates = [
                "ai/runtime/turn_orchestrator.py",
                "ai/runtime/contract_pipeline.py",
                "ai/runtime/alice_contract_factory.py",
                "ai/runtime/companion_runtime.py",
                "ai/core/route_coordinator.py",
                "ai/memory/memory_extractor.py",
                "ai/memory/personal_memory.py",
            ]
            focus = [f for f in focus_candidates if f in files][:5]
            if preview:
                if focus:
                    text = (
                        "I can inspect the workspace files. For the current Alice operator objective, highest-value files are:\n- "
                        + "\n- ".join(focus)
                        + "\nAdditional visible files:\n- "
                        + "\n- ".join(preview[:10])
                        + "\nNext best move: inspect ai/runtime/turn_orchestrator.py first."
                    )
                else:
                    text = "I can inspect these workspace files:\n- " + "\n- ".join(preview)
            else:
                text = "I do not currently see workspace files to inspect."
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
                    "Good next targets are:\n- " + "\n- ".join(focus) + "\nNext best move: inspect ai/runtime/turn_orchestrator.py because it controls route -> execute -> verify -> respond."
                )
            else:
                text = "I can attempt local inspection, but I do not currently see workspace files."
            local_execution["success"] = bool(focus)
            operator_context["awaiting_target"] = True
            return {"success": True, "response": text, "operator_context": operator_context, "local_execution": local_execution}

        if action in {"code:analyze_file", "code:read_file"}:
            target = (
                str(ctx.get("target_file") or "")
                or str(decision_meta.get("target_file") or "")
                or self._extract_target_from_query(query)
            )
            operator_context["inferred_target_file"] = target
            resolved = self.file_resolver.resolve_target(target, files) if target else {"file_exists": False, "resolved": "", "close_matches": files[:8]}
            operator_context["file_exists"] = bool(resolved["file_exists"])
            operator_context["close_matches"] = list(resolved["close_matches"])
            if not resolved["file_exists"]:
                local_execution["error"] = "target_not_found"
                local_execution["workspace_file_count"] = len(files)
                local_execution["close_matches"] = list(resolved.get("close_matches") or [])
                return {
                    "success": False,
                    "response": "",
                    "error": f"File not found: {target}" if target else "No file target provided. Ask to list workspace files.",
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
            stats = self.code_analyzer.stats(text)
            responsibility = self.code_analyzer.responsibility(rel, text)
            risks = self.code_analyzer.risk_flags(text, rel, stats)
            suggested = self.code_analyzer.suggest_next_files(rel, text, files)
            local_execution["analysis"] = {
                **stats,
                "responsibility": responsibility,
                "risk_flags": list(risks),
                "suggested_next_files": list(suggested),
            }
            local_execution["suggested_next_files"] = list(suggested)
            summary = self.response_builder.build_analysis_response(
                rel=rel,
                responsibility=responsibility,
                stats=stats,
                risks=risks,
                suggested=suggested,
            )
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

