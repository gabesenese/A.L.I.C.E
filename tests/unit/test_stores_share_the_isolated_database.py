"""Every store that lives in alice.db follows ALICE_MEMORY_DB.

The memory and goal stores already did. Alice's identity store (her opinions and
session history), the contradiction detector, the hierarchical compressor and
causal memory hard-coded data/memory/alice.db, so the test suite wrote into the
user's real database on every run, one session row per pipeline it built, and
the opinions it extracted from test replies were read back into her prompts.
"""

from pathlib import Path

import pytest

from ai.identity.identity_store import IdentityStore
from ai.memory.causal_memory import CausalMemory
from ai.memory.contradiction_detector import ContradictionDetector
from ai.memory.hierarchical_compressor import HierarchicalCompressor


def _path_of(store):
    return Path(getattr(store, "_path", None) or getattr(store, "db_path"))


@pytest.mark.parametrize("store_class", [IdentityStore, ContradictionDetector, HierarchicalCompressor, CausalMemory])
def test_the_store_opens_the_database_it_is_pointed_at(store_class, tmp_path, monkeypatch):
    target = tmp_path / "elsewhere" / "alice.db"
    target.parent.mkdir()
    monkeypatch.setenv("ALICE_MEMORY_DB", str(target))

    assert _path_of(store_class()) == target
    assert target.exists()


@pytest.mark.parametrize("store_class", [IdentityStore, ContradictionDetector, HierarchicalCompressor, CausalMemory])
def test_an_explicit_path_still_wins(store_class, tmp_path, monkeypatch):
    monkeypatch.setenv("ALICE_MEMORY_DB", str(tmp_path / "env.db"))
    explicit = tmp_path / "explicit.db"

    assert _path_of(store_class(explicit)) == explicit
