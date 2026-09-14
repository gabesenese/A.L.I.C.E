"""Reading and writing small JSON state files without losing them.

Two problems this replaces.

Pickle. Session state was written with ``pickle.dump`` and read back with
``pickle.load`` during ``ALICE.__init__``. Unpickling executes whatever the file
says to execute, so any process that could write ``data/conversation_state.pkl``
— and in the Docker image that directory is a bind mount — could run arbitrary
code the next time Alice started. The state being stored is a handful of strings
and dicts; it never needed a format that can execute.

Truncate-then-write. ``open(path, "w")`` empties the file before the new content
is written. Interrupt it — Ctrl-C, a crash, a full disk — and the file is left
empty or half-written, which is exactly when state matters most. Writing to a
temporary file and renaming means a reader sees either the old content or the
new, never a fragment.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def save_json_atomic(path: str | Path, data: Any, *, indent: Optional[int] = 2) -> bool:
    """Write ``data`` as JSON so a reader never sees a partial file.

    Returns False rather than raising: losing a state file should not take the
    turn, or the shutdown, down with it.
    """
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        # The temporary file has to share a filesystem with the target, or
        # os.replace is not atomic.
        handle, temporary = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(data, stream, indent=indent, default=str)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            return True
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
    except Exception as exc:
        logger.warning(f"Could not save state to {target}: {exc}")
        return False


def load_json(path: str | Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Read a JSON state file, returning ``default`` when it is absent or unreadable.

    Deliberately does not fall back to reading a ``.pkl`` left by an older
    version. Migrating would mean unpickling the very file this module exists to
    stop unpickling, and what is stored is a small conversational cache — losing
    one session of it costs nothing worth that.
    """
    target = Path(path)
    if not target.exists():
        return dict(default or {})
    try:
        with open(target, "r", encoding="utf-8") as stream:
            loaded = json.load(stream)
        if not isinstance(loaded, dict):
            logger.warning(f"State file {target} does not contain an object; ignoring it")
            return dict(default or {})
        return loaded
    except Exception as exc:
        logger.warning(f"Could not read state from {target}: {exc}")
        return dict(default or {})


def retire_pickle_state(path: str | Path) -> bool:
    """Rename a leftover ``.pkl`` state file aside, without reading it.

    Leaving it in place is harmless only for as long as nothing reads it, and a
    file that looks like current state invites exactly that. Returns True when a
    file was moved.
    """
    target = Path(path)
    if not target.exists():
        return False
    retired = target.with_suffix(target.suffix + ".retired")
    try:
        os.replace(target, retired)
        logger.info(f"Retired legacy pickle state {target} -> {retired.name} (not read)")
        return True
    except OSError as exc:
        logger.debug(f"Could not retire {target}: {exc}")
        return False
