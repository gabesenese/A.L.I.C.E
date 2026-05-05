from pathlib import Path

from ai.runtime.local_actions.file_index import FileIndex


class _Alice:
    PROJECT_ROOT = ""
    self_reflection = None


def test_file_index_cache_and_invalidate(tmp_path: Path):
    sample = tmp_path / "a.py"
    sample.write_text("print('x')", encoding="utf-8")
    alice = _Alice()
    index = FileIndex(alice, tmp_path, ttl_seconds=60)
    first = index.list_files()
    assert "a.py" in first
    sample.unlink()
    cached = index.list_files()
    assert "a.py" in cached
    index.invalidate()
    refreshed = index.list_files()
    assert "a.py" not in refreshed

