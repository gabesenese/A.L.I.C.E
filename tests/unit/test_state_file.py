"""Session state must survive a crash, and must not be able to run code."""

import json
import os
import pickle
from pathlib import Path

import pytest

from ai.infrastructure.state_file import load_json, retire_pickle_state, save_json_atomic


def test_a_round_trip_preserves_the_state(tmp_path):
    path = tmp_path / "state.json"
    state = {"topics": ["memory", "startup"], "turns": 12, "nested": {"a": [1, 2]}}
    assert save_json_atomic(path, state) is True
    assert load_json(path) == state


def test_a_missing_file_yields_the_default(tmp_path):
    assert load_json(tmp_path / "absent.json") == {}
    assert load_json(tmp_path / "absent.json", {"seeded": True}) == {"seeded": True}


def test_parent_directories_are_created(tmp_path):
    path = tmp_path / "deep" / "nested" / "state.json"
    assert save_json_atomic(path, {"ok": True}) is True
    assert path.exists()


def test_an_interrupted_write_leaves_the_previous_state_intact(tmp_path, monkeypatch):
    """open(path, "w") empties the file before writing, so a crash mid-save left
    it blank — which reads back next start as a user with no state at all."""
    path = tmp_path / "state.json"
    save_json_atomic(path, {"generation": 1})

    real_replace = os.replace

    def failing_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", failing_replace)
    assert save_json_atomic(path, {"generation": 2}) is False
    monkeypatch.setattr(os, "replace", real_replace)

    assert load_json(path) == {"generation": 1}


def test_a_failed_write_leaves_no_temporary_files_behind(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    monkeypatch.setattr(os, "replace", lambda src, dst: (_ for _ in ()).throw(OSError("nope")))
    save_json_atomic(path, {"generation": 1})
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


def test_unserialisable_values_do_not_raise(tmp_path):
    """Losing a state file must not take the shutdown down with it."""
    path = tmp_path / "state.json"
    assert save_json_atomic(path, {"when": object()}) is True
    assert isinstance(load_json(path)["when"], str)


def test_corrupt_json_reads_as_absent_rather_than_raising(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{ this is not json", encoding="utf-8")
    assert load_json(path) == {}


def test_a_json_file_that_is_not_an_object_is_ignored(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    assert load_json(path) == {}


# -- the pickle the format replaces ------------------------------------------


class _Exploding:
    """Stands in for a malicious payload: unpickling runs __reduce__."""

    def __reduce__(self):
        return (os.system, ("echo pwned",))


def test_a_pickle_left_by_an_older_version_is_never_read(tmp_path):
    """Migrating old state would mean unpickling the very file this module
    exists to stop unpickling. The state is a small conversational cache;
    losing one session of it costs less than executing it."""
    pickle_path = tmp_path / "conversation_state.pkl"
    pickle_path.write_bytes(pickle.dumps(_Exploding()))

    assert retire_pickle_state(pickle_path) is True
    assert not pickle_path.exists()
    assert (tmp_path / "conversation_state.pkl.retired").exists()


def test_retiring_an_absent_pickle_is_a_no_op(tmp_path):
    assert retire_pickle_state(tmp_path / "nothing.pkl") is False


def test_load_json_refuses_a_pickle_payload(tmp_path):
    path = tmp_path / "state.json"
    path.write_bytes(pickle.dumps({"harmless": True}))
    assert load_json(path) == {}


def _source_of(obj):
    import inspect

    return inspect.getsource(obj)


def test_alice_no_longer_unpickles_state_during_startup():
    """ALICE.__init__ called pickle.load on data/conversation_state.pkl, and in
    the Docker image data/ is a bind mount — anything that could write there
    could run code on the next start."""
    from app.main import ALICE

    source = _source_of(ALICE._load_conversation_state)
    assert "pickle.load" not in source
    assert "retire_pickle_state" in source


@pytest.mark.parametrize("method", ["load_state", "save_state", "_load_context", "save_context"])
def test_the_context_engine_no_longer_pickles_its_state(method):
    from ai.memory.context_engine import ContextEngine

    source = _source_of(getattr(ContextEngine, method))
    assert "pickle.load" not in source
    assert "pickle.dump" not in source


def test_conversation_state_is_written_as_readable_json(tmp_path):
    """A state file a person can open and read is one they can also fix."""
    path = tmp_path / "conversation_state.json"
    save_json_atomic(path, {"conversation_topics": ["alice", "memory"]})
    text = Path(path).read_text(encoding="utf-8")
    assert "conversation_topics" in text
    assert "\n" in text  # indented, not a single line
