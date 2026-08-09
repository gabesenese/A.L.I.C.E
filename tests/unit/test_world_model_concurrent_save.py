"""WorldModel.save must survive concurrent writers.

save() staged every write to a single fixed `.tmp` path and then renamed it, so two
writers raced on the same staging file. On Windows the rename additionally fails with
PermissionError whenever anything else holds the target open. The result was an
intermittent crash mid-turn and a suite that failed roughly one run in eight.
"""

import json
import threading

from memory.world_model import WorldModel


def test_concurrent_saves_all_succeed_and_leave_valid_json(tmp_path):
    path = tmp_path / "world_model.json"
    models = [WorldModel(path=path) for _ in range(8)]
    errors = []
    barrier = threading.Barrier(len(models))

    def save(model, index):
        try:
            barrier.wait()
            for _ in range(5):
                model.record_topic_mentions(f"writer {index} mentioned the routing refactor")
                model.save()
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=save, args=(model, i)) for i, model in enumerate(models)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, errors
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] >= 1


def test_save_leaves_no_temp_files_behind(tmp_path):
    path = tmp_path / "world_model.json"
    model = WorldModel(path=path)
    model.save()
    model.save()

    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []
