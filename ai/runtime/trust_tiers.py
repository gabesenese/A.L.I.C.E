"""Decides what Alice may do unattended, what needs a yes, and what is never allowed.

Asking permission for every action is not an assistant, it is a confirmation dialog
with a personality. Asking for none is not something you leave running. The split
is by blast radius: reads and edits inside the workspace are reversible from a git
checkpoint, so they run. Anything that leaves the workspace, reaches another
person, or cannot be undone stops and asks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ai.core import tool_catalog as catalog

TIER_AUTO = "auto"
TIER_CONFIRM = "confirm"
TIER_REFUSE = "refuse"

_PATH_ARGUMENTS = ("path", "file", "target", "subdirectory")

# Shell shapes that are never run, regardless of tier or approval.
_REFUSED_COMMAND_PATTERNS = (
    r"\brm\s+-[a-z]*[rf]",
    r"\brmdir\s+/s",
    r"\bdel\s+/[fqs]",
    r"\bformat\b",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bgit\s+push\b.*--force",
    r"\bgit\s+reset\s+--hard\b",
    r"\bcurl\b.*\|\s*(?:ba)?sh",
    r"\bchmod\s+777\b",
    r":\(\)\{.*\};:",
)

# Commands allowed to run unattended. Everything else needs a yes.
_ALLOWED_COMMANDS = (
    "pytest",
    "python -m pytest",
    "ruff check",
    "ruff format",
    "git status",
    "git diff",
    "git log",
    "git branch",
)

_REFUSED_RE = re.compile("|".join(_REFUSED_COMMAND_PATTERNS), re.IGNORECASE)


@dataclass(frozen=True)
class TierDecision:
    tier: str
    reason: str
    scope: str = ""
    summary: str = ""

    @property
    def allowed_unattended(self) -> bool:
        return self.tier == TIER_AUTO

    def to_dict(self) -> Dict[str, Any]:
        return {"tier": self.tier, "reason": self.reason, "scope": self.scope, "summary": self.summary}


def _paths_in(arguments: Dict[str, Any]) -> list[str]:
    return [str(arguments[key]) for key in _PATH_ARGUMENTS if str(arguments.get(key) or "").strip()]


def _all_paths_inside_workspace(arguments: Dict[str, Any]) -> bool:
    paths = _paths_in(arguments)
    if not paths:
        return True
    return all(catalog._resolve_inside_project(p) is not None for p in paths)


def _command_is_refused(command: str) -> bool:
    return bool(_REFUSED_RE.search(str(command or "")))


def _command_is_allowlisted(command: str) -> bool:
    normalized = " ".join(str(command or "").strip().lower().split())
    return any(normalized.startswith(allowed) for allowed in _ALLOWED_COMMANDS)


def classify(tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> TierDecision:
    args = dict(arguments or {})
    spec = catalog.get_spec(tool_name)
    if spec is None:
        return TierDecision(tier=TIER_REFUSE, reason="unknown_tool", summary=f"{tool_name} is not a known tool")

    if tool_name == "run_command":
        command = str(args.get("command") or "")
        if _command_is_refused(command):
            return TierDecision(TIER_REFUSE, "destructive_command", summary=command)
        if _command_is_allowlisted(command):
            return TierDecision(TIER_AUTO, "allowlisted_command", scope="shell", summary=command)
        return TierDecision(TIER_CONFIRM, "command_not_allowlisted", scope="shell", summary=command)

    if spec.risk == catalog.RISK_READ:
        return TierDecision(TIER_AUTO, "read_only", scope=spec.name)

    if spec.risk == catalog.RISK_OUTWARD:
        return TierDecision(
            TIER_CONFIRM,
            "leaves_this_machine",
            scope=spec.name,
            summary=f"{spec.name} {args}",
        )

    # Replacing a file that already exists destroys whatever the model did not
    # reproduce. It asked for that explicitly, so it stops and asks the user too.
    if tool_name == "write_workspace_file" and args.get("overwrite"):
        target = catalog._resolve_inside_project(str(args.get("path") or ""))
        if target is not None and target.is_file():
            return TierDecision(
                TIER_CONFIRM,
                "overwrites_existing_file",
                scope=spec.name,
                summary=str(args.get("path") or ""),
            )

    if not _all_paths_inside_workspace(args):
        return TierDecision(
            TIER_CONFIRM,
            "writes_outside_workspace",
            scope=spec.name,
            summary=", ".join(_paths_in(args)),
        )

    return TierDecision(TIER_AUTO, "reversible_workspace_write", scope=spec.name, summary=", ".join(_paths_in(args)))
