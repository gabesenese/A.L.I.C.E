"""Typed tool surface the model selects from, in the schema Ollama expects.

Tool choice used to be keyword matching over a fixed domain table, so a request the
table did not anticipate produced an invented answer rather than a tool call. The
model now receives real tool definitions and picks one, and every call is dispatched
through a named handler or the existing plugin execution path.

Each spec carries a risk tier. Nothing here consumes it yet; the graduated autonomy
work reads it to decide what runs unattended and what needs confirmation.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ai.infrastructure.paths import project_root

RISK_READ = "read"
RISK_WRITE = "write"
RISK_OUTWARD = "outward"

MAX_FILE_CHARS = 8000
MAX_LIST_RESULTS = 200
MAX_SEARCH_RESULTS = 40

_SKIP_DIRECTORIES = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    ".mypy_cache",
}
_SOURCE_GLOBS = ("*.py", "*.md", "*.toml", "*.json", "*.yml", "*.yaml", "*.txt", "*.bat", "*.sh")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]
    risk: str = RISK_READ
    handler: Optional[Callable[..., Dict[str, Any]]] = None
    intent: str = ""
    query_from: str = ""
    entity_map: Dict[str, str] = field(default_factory=dict)

    def to_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolExecution:
    tool: str
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool,
            "success": self.success,
            "data": self.data,
            "summary": self.summary,
            "error": self.error,
        }


def _resolve_inside_project(candidate: str) -> Optional[Path]:
    """Resolve a path and return it only if it lands inside the project.

    An absolute path must never be quietly reinterpreted as a relative one: stripping
    the leading separator turned '/etc/passwd' into '<project>/etc/passwd', which
    reads as contained while hiding that the caller asked to leave the workspace.
    """
    root = project_root().resolve()
    raw = str(candidate or "").strip()
    if not raw:
        return None
    try:
        path = Path(raw)
        rooted = path.is_absolute() or raw.startswith(("/", "\\"))
        target = path.resolve() if rooted else (root / path).resolve()
    except (OSError, ValueError):
        return None
    if target == root or root in target.parents:
        return target
    return None


def _iter_source_files(scope: Path) -> List[Path]:
    found: List[Path] = []
    for path in scope.rglob("*"):
        if not path.is_file():
            continue
        if any(part in _SKIP_DIRECTORIES for part in path.parts):
            continue
        if not any(fnmatch.fnmatch(path.name, pattern) for pattern in _SOURCE_GLOBS):
            continue
        found.append(path)
    return found


def _list_workspace_files(subdirectory: str = "", limit: int = MAX_LIST_RESULTS) -> Dict[str, Any]:
    root = project_root().resolve()
    scope = _resolve_inside_project(subdirectory) if subdirectory else root
    if scope is None or not scope.exists():
        return {"success": False, "error": f"No such directory in the workspace: {subdirectory}"}

    limit = max(1, min(int(limit or MAX_LIST_RESULTS), MAX_LIST_RESULTS))
    files = sorted(str(p.relative_to(root)).replace("\\", "/") for p in _iter_source_files(scope))
    return {
        "success": True,
        "scope": str(scope.relative_to(root)).replace("\\", "/") if scope != root else ".",
        "total_files": len(files),
        "files": files[:limit],
        "truncated": len(files) > limit,
    }


def _read_workspace_file(path: str, max_chars: int = MAX_FILE_CHARS) -> Dict[str, Any]:
    target = _resolve_inside_project(path)
    if target is None:
        return {"success": False, "error": f"Path is outside the workspace: {path}"}
    if not target.is_file():
        return {"success": False, "error": f"No such file in the workspace: {path}"}

    max_chars = max(200, min(int(max_chars or MAX_FILE_CHARS), MAX_FILE_CHARS))
    text = target.read_text(encoding="utf-8", errors="replace")
    root = project_root().resolve()
    return {
        "success": True,
        "path": str(target.relative_to(root)).replace("\\", "/"),
        "line_count": text.count("\n") + 1,
        "content": text[:max_chars],
        "truncated": len(text) > max_chars,
    }


def _write_workspace_file(path: str, content: str, overwrite: bool = False) -> Dict[str, Any]:
    target = _resolve_inside_project(path)
    if target is None:
        return {"success": False, "error": f"Path is outside the workspace: {path}"}

    root = project_root().resolve()
    existed = target.is_file()
    if existed and not overwrite:
        return {
            "success": False,
            "error": (
                f"{path} already exists. Use edit_workspace_file to change part of it, "
                "or pass overwrite=true to replace the whole file deliberately."
            ),
        }
    previous = target.read_text(encoding="utf-8", errors="replace") if existed else ""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(str(content or ""), encoding="utf-8")
    return {
        "success": True,
        "path": str(target.relative_to(root)).replace("\\", "/"),
        "created": not existed,
        "bytes_written": len(str(content or "").encode("utf-8")),
        "previous_line_count": previous.count("\n") + 1 if existed else 0,
    }


def _edit_workspace_file(path: str, find: str, replace: str) -> Dict[str, Any]:
    target = _resolve_inside_project(path)
    if target is None:
        return {"success": False, "error": f"Path is outside the workspace: {path}"}
    if not target.is_file():
        return {"success": False, "error": f"No such file in the workspace: {path}"}

    original = target.read_text(encoding="utf-8", errors="replace")
    occurrences = original.count(str(find or ""))
    if not str(find or ""):
        return {"success": False, "error": "The text to find must not be empty."}
    if occurrences == 0:
        return {"success": False, "error": "That exact text was not found in the file."}
    if occurrences > 1:
        return {
            "success": False,
            "error": f"That text appears {occurrences} times. Provide more surrounding context so it is unique.",
        }

    root = project_root().resolve()
    target.write_text(original.replace(str(find), str(replace or ""), 1), encoding="utf-8")
    return {
        "success": True,
        "path": str(target.relative_to(root)).replace("\\", "/"),
        "replacements": 1,
    }


def _run_command(command: str, timeout: int = 120) -> Dict[str, Any]:
    import subprocess

    text = str(command or "").strip()
    if not text:
        return {"success": False, "error": "A command is required."}
    try:
        completed = subprocess.run(
            text,
            shell=True,
            cwd=str(project_root()),
            capture_output=True,
            text=True,
            timeout=max(1, min(int(timeout or 120), 600)),
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Command timed out: {text}"}
    except Exception as exc:
        return {"success": False, "error": str(exc)}

    return {
        "success": completed.returncode == 0,
        "command": text,
        "exit_code": completed.returncode,
        "stdout": (completed.stdout or "")[-MAX_FILE_CHARS:],
        "stderr": (completed.stderr or "")[-2000:],
        "error": "" if completed.returncode == 0 else f"exit code {completed.returncode}",
    }


def _search_workspace(query: str, subdirectory: str = "", limit: int = MAX_SEARCH_RESULTS) -> Dict[str, Any]:
    needle = str(query or "").strip()
    if not needle:
        return {"success": False, "error": "A non-empty query is required."}

    root = project_root().resolve()
    scope = _resolve_inside_project(subdirectory) if subdirectory else root
    if scope is None or not scope.exists():
        return {"success": False, "error": f"No such directory in the workspace: {subdirectory}"}

    limit = max(1, min(int(limit or MAX_SEARCH_RESULTS), MAX_SEARCH_RESULTS))
    lowered = needle.lower()
    matches: List[Dict[str, Any]] = []
    for path in _iter_source_files(scope):
        try:
            for number, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                if lowered in line.lower():
                    matches.append(
                        {
                            "path": str(path.relative_to(root)).replace("\\", "/"),
                            "line": number,
                            "text": line.strip()[:200],
                        }
                    )
                    if len(matches) >= limit:
                        break
        except OSError:
            continue
        if len(matches) >= limit:
            break

    return {"success": True, "query": needle, "match_count": len(matches), "matches": matches}


CATALOG: List[ToolSpec] = [
    ToolSpec(
        name="list_workspace_files",
        description=(
            "List real source files in the A.L.I.C.E project on this machine. "
            "Use this whenever asked what files exist, what you can inspect, or what the project contains. "
            "Never guess or recall filenames from memory: call this and report exactly what it returns."
        ),
        parameters={
            "type": "object",
            "properties": {
                "subdirectory": {
                    "type": "string",
                    "description": "Optional path to scope the listing, for example 'ai/runtime'.",
                },
                "limit": {"type": "integer", "description": "Maximum number of files to return."},
            },
            "required": [],
        },
        handler=_list_workspace_files,
    ),
    ToolSpec(
        name="read_workspace_file",
        description=(
            "Read the real contents of one file in the project. "
            "Use before describing, reviewing, or explaining any file. Never describe a file you have not read."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Project relative path, for example 'ai/runtime/agent_loop.py'.",
                },
                "max_chars": {"type": "integer", "description": "Maximum characters to return."},
            },
            "required": ["path"],
        },
        handler=_read_workspace_file,
    ),
    ToolSpec(
        name="search_workspace",
        description=(
            "Search the project source for a literal string and return matching files with line numbers. "
            "Use to locate where something is defined or used before answering."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Literal text to search for."},
                "subdirectory": {"type": "string", "description": "Optional path to scope the search."},
                "limit": {"type": "integer", "description": "Maximum number of matches to return."},
            },
            "required": ["query"],
        },
        handler=_search_workspace,
    ),
    ToolSpec(
        name="write_workspace_file",
        description=(
            "Create a NEW file in the project. This refuses to touch a file that already exists, "
            "because writing partial content over an existing file destroys the rest of it. "
            "To change an existing file use edit_workspace_file. Only pass overwrite when you "
            "intend to replace an entire file and have its full new contents."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Project relative path to write."},
                "content": {"type": "string", "description": "Full file contents."},
                "overwrite": {
                    "type": "boolean",
                    "description": "Set true only to deliberately replace an entire existing file.",
                },
            },
            "required": ["path", "content"],
        },
        risk=RISK_WRITE,
        handler=_write_workspace_file,
    ),
    ToolSpec(
        name="edit_workspace_file",
        description=(
            "Replace one exact block of text in an existing project file. "
            "The text to find must appear exactly once, so include enough surrounding context."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Project relative path to edit."},
                "find": {"type": "string", "description": "Exact text to replace, unique in the file."},
                "replace": {"type": "string", "description": "Replacement text."},
            },
            "required": ["path", "find", "replace"],
        },
        risk=RISK_WRITE,
        handler=_edit_workspace_file,
    ),
    ToolSpec(
        name="run_command",
        description=(
            "Run a shell command in the project directory and return its output. "
            "Use for tests, linters, and git inspection, for example 'pytest -q' or 'git status'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The command line to run."},
                "timeout": {"type": "integer", "description": "Seconds to wait before giving up."},
            },
            "required": ["command"],
        },
        risk=RISK_WRITE,
        handler=_run_command,
    ),
    ToolSpec(
        name="get_system_status",
        description="Read live CPU, memory, and disk usage from this machine. Use for any question about system health.",
        parameters={"type": "object", "properties": {}, "required": []},
        intent="system:status",
        query_from="system status",
    ),
    ToolSpec(
        name="get_current_weather",
        description="Get current weather conditions for a location. Use for any question about weather right now.",
        parameters={
            "type": "object",
            "properties": {"location": {"type": "string", "description": "City name, for example 'Toronto'."}},
            "required": [],
        },
        intent="weather:current",
        query_from="weather",
        entity_map={"location": "location"},
    ),
    ToolSpec(
        name="list_notes",
        description="List the user's saved notes. Use before claiming anything about what the user has written down.",
        parameters={"type": "object", "properties": {}, "required": []},
        intent="notes:list",
        query_from="list my notes",
    ),
    ToolSpec(
        name="search_notes",
        description="Search the user's saved notes for a term, and report only what is returned.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Term to search notes for."}},
            "required": ["query"],
        },
        intent="notes:search",
        query_from="search notes for {query}",
        entity_map={"query": "search_term"},
    ),
    ToolSpec(
        name="create_note",
        description="Save a new note for the user.",
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Short title for the note."},
                "content": {"type": "string", "description": "Body text of the note."},
            },
            "required": ["title"],
        },
        risk=RISK_WRITE,
        intent="notes:create",
        query_from="create a note titled {title}",
        entity_map={"title": "title", "content": "content"},
    ),
]

_BY_NAME: Dict[str, ToolSpec] = {spec.name: spec for spec in CATALOG}


def get_spec(name: str) -> Optional[ToolSpec]:
    return _BY_NAME.get(str(name or "").strip())


def tool_names() -> List[str]:
    return [spec.name for spec in CATALOG]


def build_tool_schemas(names: Optional[List[str]] = None, max_risk: str = RISK_OUTWARD) -> List[Dict[str, Any]]:
    """Return Ollama tool definitions, optionally narrowed to a subset."""
    allowed_risk = {RISK_READ}
    if max_risk in (RISK_WRITE, RISK_OUTWARD):
        allowed_risk.add(RISK_WRITE)
    if max_risk == RISK_OUTWARD:
        allowed_risk.add(RISK_OUTWARD)

    wanted = set(names or tool_names())
    return [spec.to_schema() for spec in CATALOG if spec.name in wanted and spec.risk in allowed_risk]


def sanitize_arguments(spec: ToolSpec, raw: Any) -> Dict[str, Any]:
    """Drop nulls and unknown keys the model may emit alongside a tool call."""
    if not isinstance(raw, dict):
        return {}
    allowed = set((spec.parameters or {}).get("properties", {}))
    return {key: value for key, value in raw.items() if key in allowed and value is not None and value != ""}


def execute_tool(
    name: str,
    arguments: Any = None,
    *,
    plugin_manager: Any = None,
    context: Optional[Dict[str, Any]] = None,
) -> ToolExecution:
    spec = get_spec(name)
    if spec is None:
        return ToolExecution(tool=str(name), success=False, error=f"Unknown tool: {name}")

    args = sanitize_arguments(spec, arguments)
    missing = [key for key in (spec.parameters or {}).get("required", []) if key not in args]
    if missing:
        return ToolExecution(
            tool=spec.name,
            success=False,
            error=f"Missing required argument(s): {', '.join(missing)}",
        )

    if spec.handler is not None:
        try:
            result = spec.handler(**args)
        except Exception as exc:
            return ToolExecution(tool=spec.name, success=False, error=str(exc))
        success = bool(result.get("success", True))
        return ToolExecution(
            tool=spec.name,
            success=success,
            data=result,
            summary=_summarize(spec.name, result),
            error=str(result.get("error") or "") if not success else "",
        )

    if plugin_manager is None:
        return ToolExecution(tool=spec.name, success=False, error="No plugin manager available to run this tool.")

    entities = {spec.entity_map[key]: value for key, value in args.items() if key in spec.entity_map}
    query = spec.query_from.format(**{key: args.get(key, "") for key in args}) if spec.query_from else spec.name
    try:
        raw = plugin_manager.execute_for_intent(spec.intent, query, entities, dict(context or {}))
    except Exception as exc:
        return ToolExecution(tool=spec.name, success=False, error=str(exc))

    if not isinstance(raw, dict):
        return ToolExecution(tool=spec.name, success=False, error="Plugin returned no usable result.")

    success = bool(raw.get("success", False))
    return ToolExecution(
        tool=spec.name,
        success=success,
        data=raw,
        summary=str(raw.get("response") or "").strip(),
        error=str(raw.get("error") or raw.get("message") or "") if not success else "",
    )


def _summarize(tool: str, result: Dict[str, Any]) -> str:
    if not result.get("success", True):
        return str(result.get("error") or "")
    if tool == "list_workspace_files":
        return f"{result.get('total_files', 0)} files under {result.get('scope', '.')}"
    if tool == "read_workspace_file":
        return f"{result.get('path', '')} ({result.get('line_count', 0)} lines)"
    if tool == "search_workspace":
        return f"{result.get('match_count', 0)} matches for {result.get('query', '')}"
    return ""
