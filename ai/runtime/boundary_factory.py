"""Thin boundary factory wrapper.

This module exists so startup wiring can import a smaller surface while
`alice_contract_factory` remains backward-compatible.
"""

from __future__ import annotations

from typing import Any

from ai.contracts import RuntimeBoundaries
from ai.runtime.alice_contract_factory import build_runtime_boundaries


def build_boundaries(alice: Any) -> RuntimeBoundaries:
    return build_runtime_boundaries(alice)

