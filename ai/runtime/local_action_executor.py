"""Local runtime actions for operator-style code inspection."""

from __future__ import annotations

import os
import re
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path
from typing import Any, Dict, List


class LocalActionExecutor:
    def __init__(self, alice: Any):
        self.alice = alice
        self.root = self._resolve_root()

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

    def _safe_rel(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root)).replace("\\", "/")
        except Exception:
            return str(path).replace("\\", "/")

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
                    rel = self._safe_rel(Path(dirpath, name))
                    out.append(rel)
                    if len(out) >= limit:
                        return out
        return out

    def _resolve_target(self, target: str, files: List[str]) -> Dict[str, Any]:
        target = str(target or "").strip().strip("`'\"").replace("\\", "/").lstrip("./")
        if not target:
            return {"file_exists": False, "resolved": "", "close_matches": files[:8]}
        exact = [f for f in files if f.lower() == target.lower()]
        if exact:
            return {"file_exists": True, "resolved": exact[0], "close_matches": []}
        suffix = [f for f in files if f.lower().endswith("/" + target.lower())]
        if suffix:
            return {"file_exists": True, "resolved": suffix[0], "close_matches": []}
        basename_exact = [f for f in files if Path(f).name.lower() == Path(target).name.lower()]
        if len(basename_exact) == 1:
            return {"file_exists": True, "resolved": basename_exact[0], "close_matches": []}
        if len(basename_exact) > 1:
            ranked_basename = sorted(
                basename_exact,
                key=lambda candidate: SequenceMatcher(None, target.lower(), candidate.lower()).ratio(),
                reverse=True,
            )
            return {"file_exists": False, "resolved": "", "close_matches": ranked_basename[:8], "ambiguous": True}
        basename = target.split("/")[-1].lower()
        contains = [f for f in files if basename and basename in Path(f).name.lower()]
        fuzzy_basename = get_close_matches(basename, [Path(f).name.lower() for f in files], n=8, cutoff=0.55)
        fuzzy_files: List[str] = []
        for item in files:
            file_name = Path(item).name.lower()
            if file_name in fuzzy_basename and item not in fuzzy_files:
                fuzzy_files.append(item)
        ranked = sorted(
            set(contains + fuzzy_files),
            key=lambda candidate: SequenceMatcher(None, target.lower(), candidate.lower()).ratio(),
            reverse=True,
        )
        close = ranked[:8]
        return {"file_exists": False, "resolved": "", "close_matches": close}

    def _read_file_text(self, rel_path: str, max_chars: int = 12000) -> str:
        p = self.root / rel_path
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(max_chars)

    def _extract_target_from_query(self, query: str) -> str:
        m = re.search(r"([a-zA-Z0-9_./\\-]+\.[a-zA-Z0-9]{1,8})\b", str(query or ""))
        if m:
            return m.group(1)
        return ""

    def _responsibility(self, rel: str, text: str) -> str:
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

    def _stats(self, text: str) -> Dict[str, Any]:
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

    def _risk_flags(self, text: str, rel: str, stats: Dict[str, Any]) -> List[str]:
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

    def _suggest_next_files(self, rel: str, text: str, files: List[str]) -> List[str]:
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

    def execute(self, *, action: str, query: str, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        ctx = dict(context or {})
        files = self._list_files()
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
            resolved = self._resolve_target(target, files) if target else {"file_exists": False, "resolved": "", "close_matches": files[:8]}
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
            stats = self._stats(text)
            responsibility = self._responsibility(rel, text)
            risks = self._risk_flags(text, rel, stats)
            suggested = self._suggest_next_files(rel, text, files)
            local_execution["analysis"] = {
                **stats,
                "responsibility": responsibility,
                "risk_flags": list(risks),
                "suggested_next_files": list(suggested),
            }
            local_execution["suggested_next_files"] = list(suggested)
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
            summary = " ".join(parts)
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
