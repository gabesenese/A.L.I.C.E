"""Filesystem locations resolved against the project root rather than the current directory."""

from __future__ import annotations

import os
from pathlib import Path

_TRUTHY = {"1", "true", "yes", "on"}


def project_root() -> Path:
    override = os.environ.get("ALICE_PROJECT_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def credential_path(filename: str) -> Path:
    return project_root() / "config" / "cred" / filename


def interactive_auth_allowed() -> bool:
    return os.environ.get("ALICE_ALLOW_INTERACTIVE_AUTH", "").strip().lower() in _TRUTHY
