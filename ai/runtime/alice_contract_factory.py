"""Compatibility wrapper for runtime boundary construction."""

from __future__ import annotations

from typing import Any

from ai.contracts import RuntimeBoundaries
from ai.runtime.boundaries.boundary_factory import build_runtime_boundaries as _impl


def build_runtime_boundaries(alice: Any) -> RuntimeBoundaries:
    return _impl(alice)


__all__ = ["build_runtime_boundaries"]

