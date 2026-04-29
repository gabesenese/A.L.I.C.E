"""Lightweight task classification for model routing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TaskProfile:
    task_type: str
    multi_step: bool


_CODE_KW = {
    "code",
    "python",
    "bug",
    "debug",
    "stack trace",
    "traceback",
    "refactor",
    "unit test",
    "pytest",
    "function",
    "class",
    "compile",
    "syntax",
}
_PLAN_KW = {"plan", "roadmap", "strategy", "architecture", "design", "steps"}
_MULTI_KW = {" and ", " then ", " also ", " after that ", " plus "}


def classify_task(request: str) -> TaskProfile:
    text = f" {str(request or '').lower()} "
    multi_step = any(k in text for k in _MULTI_KW)

    if any(k in text for k in _CODE_KW):
        return TaskProfile(task_type="coding", multi_step=multi_step)
    if any(k in text for k in _PLAN_KW):
        return TaskProfile(task_type="planning", multi_step=True)
    if any(q in text for q in ("how", "why", "what if", "compare", "tradeoff")):
        return TaskProfile(task_type="reasoning", multi_step=multi_step)
    return TaskProfile(task_type="simple", multi_step=multi_step)
