"""Shared pytest fixtures for integration tests."""

import atexit
import logging
import os
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault("ALICE_ENABLE_BACKGROUND_SERVICES", "0")
# The startup doctor resolves the checkout by absolute path and writes its health
# summary into data/qa there, so merely booting the app in a test wrote to the
# user's own data directory. Its own tests build a StartupDoctor against a
# tmp_path, so nothing here needs the real one to run.
os.environ.setdefault("ALICE_STARTUP_DOCTOR", "0")
# Every store sharing data/memory/alice.db reads ALICE_MEMORY_DB. Each test gets
# its own below; this covers everything outside a test. Every ContractPipeline
# registers an exit handler that closes its session in the identity store, and
# those run after per-test isolation is undone, so the suite wrote a session row
# into the user's real database for each pipeline it built.
_SESSION_DB_DIR = tempfile.mkdtemp(prefix="alice-tests-")
os.environ["ALICE_MEMORY_DB"] = str(Path(_SESSION_DB_DIR) / "alice.db")
# Behaviour events, audits and improvement hypotheses land here; the default is data/.
os.environ["ALICE_SELF_IMPROVEMENT_DATA_DIR"] = str(Path(_SESSION_DB_DIR) / "self_improvement")
atexit.register(shutil.rmtree, _SESSION_DB_DIR, True)

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from ai.plugins.notes_plugin import NotesManager, NotesPlugin
from ai.runtime.contract_pipeline import ContractPipeline

from app.main import app
from app.api.dependencies import get_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def pytest_configure(config):
    # torch registers dump_cache_stats() via @atexit.register at module import.
    # After pytest closes its log handlers, dump_cache_stats() tries to write
    # to a closed stream and prints "--- Logging error --- ValueError: I/O
    # operation on closed file." to the console on every test run.
    # Silencing the logger and unregistering the atexit hook eliminates both.
    logging.getLogger("torch._subclasses.fake_tensor").setLevel(logging.CRITICAL)
    try:
        from torch._subclasses import fake_tensor

        atexit.unregister(fake_tensor.dump_cache_stats)
    except Exception:
        pass


@pytest.fixture(scope="session", autouse=True)
def restore_data_directory():
    """Return data/ to its pre-session contents when the run finishes.

    Tests write learned state, journals, and goals into data/, so each run started
    from whatever the previous run left behind. Test order is fixed, yet the set of
    failures changed between identical runs, because the suite was effectively
    iterating on its own leftover state. Restoring afterwards makes every run start
    from the same baseline, so a failure means the same thing twice.
    """
    if not DATA_DIR.exists():
        yield
        return

    snapshot_root = Path(tempfile.mkdtemp(prefix="alice-data-snapshot-"))
    snapshot = snapshot_root / "data"
    shutil.copytree(DATA_DIR, snapshot)
    original = {p.relative_to(snapshot) for p in snapshot.rglob("*") if p.is_file()}
    try:
        yield
    finally:
        for relative in original:
            target = DATA_DIR / relative
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(snapshot / relative, target)
            except OSError:
                pass
        for path in list(DATA_DIR.rglob("*")):
            if path.is_file() and path.relative_to(DATA_DIR) not in original:
                try:
                    path.unlink()
                except OSError:
                    pass
        shutil.rmtree(snapshot_root, ignore_errors=True)


@pytest.fixture(autouse=True)
def isolate_memory_store(tmp_path, monkeypatch):
    """Give every test its own memory database.

    SQLiteMemoryStore's path was a bare constant, so the whole suite wrote to
    data/memory/alice.db — the user's real memories. Two consequences, both
    observed. Tests mutated live user data on every run. And several pytest
    workers plus a background agent hitting one SQLite file produced "database
    disk image is malformed", after which MemorySystem._load_memories catches the
    error and the process runs with recall silently disabled — which is also how
    a test that passes alone fails in a full run.
    """
    import ai.goals.goal_store as goal_store
    import ai.identity.identity_store as identity_store
    import ai.memory.causal_memory as causal_memory
    import ai.memory.contradiction_detector as contradiction_detector
    import ai.memory.hierarchical_compressor as hierarchical_compressor
    import ai.memory.memory_store as memory_store

    monkeypatch.setenv("ALICE_MEMORY_DB", str(tmp_path / "alice.db"))
    # The improvement loop reads its own history to decide what recurs, so one
    # test's failures must not become the next test's pattern.
    monkeypatch.setenv("ALICE_SELF_IMPROVEMENT_DATA_DIR", str(tmp_path / "self_improvement"))
    monkeypatch.setattr(memory_store, "_memory_store", None, raising=False)
    # Alice's own opinions and session history live in the same file, and her
    # opinions are read back into every prompt.
    monkeypatch.setattr(identity_store, "_store", None, raising=False)
    monkeypatch.setattr(contradiction_detector, "_detector", None, raising=False)
    monkeypatch.setattr(hierarchical_compressor, "_compressor", None, raising=False)
    monkeypatch.setattr(causal_memory, "_causal_memory", None, raising=False)
    # GoalStore writes goals into the same file, behind its own singleton, so
    # leaving it alone means the goal stack Gabriel is actually working from
    # accumulates whatever strings the suite feeds through a turn.
    monkeypatch.setattr(goal_store, "_store", None, raising=False)
    yield
    memory_store._memory_store = None
    goal_store._store = None


@pytest.fixture(autouse=True)
def isolate_project_memory(tmp_path, monkeypatch):
    """Give every test its own project memory store.

    data/project_memory.json is keyed by user id and persists between runs, so a
    test that drives a turn leaves operator state, recommendations, and inspected
    files behind for whatever runs next. That made the suite order dependent: the
    set of failures changed between identical runs while each test passed alone.
    """
    import ai.memory.project_memory as project_memory

    monkeypatch.setattr(project_memory, "PROJECT_MEMORY_PATH", tmp_path / "project_memory.json")


@pytest.fixture(autouse=True)
def reset_routing_confidence_singletons(tmp_path, monkeypatch):
    """Isolate the learned signals that shift routing confidence between tests.

    Behavioral priors and intent success rates reach routing through
    process-wide singletons backed by files under data/, so the turns one test
    drives raise the priors the next test routes under. That moved borderline
    turns across a decision band and made the suite order dependent: tests
    passed alone and failed in the same file. Clearing the singletons is not
    enough on its own — they reload the same accumulated profile from disk — so
    each test also gets its own profile directory.
    """
    import ai.core.confidence_fusion as confidence_fusion
    import ai.learning.user_profile_engine as user_profile_engine
    import ai.optimization.clarification_feedback_loop as clarification_feedback_loop

    def _clear():
        user_profile_engine._profile_engine = None
        clarification_feedback_loop._loop = None
        confidence_fusion._fusion = None
        confidence_fusion.ConfidenceFusion._rates_cache = {}
        confidence_fusion.ConfidenceFusion._rates_stamp = None

    _clear()
    profiles_dir = tmp_path / "user_profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    real_get = user_profile_engine.get_profile_engine
    monkeypatch.setattr(
        user_profile_engine,
        "get_profile_engine",
        lambda storage_path=str(profiles_dir): real_get(storage_path=storage_path),
    )
    yield
    _clear()


@pytest.fixture
def plugin(tmp_path):
    notes_dir = tmp_path / "notes"
    notes_plugin = NotesPlugin()
    notes_plugin.manager = NotesManager(notes_dir=str(notes_dir))
    notes_plugin.last_note_id = None
    notes_plugin.last_note_title = None
    notes_plugin.last_note_result_ids = []
    notes_plugin.learning_state_path = tmp_path / "notes_learning_state.json"
    notes_plugin.telemetry_log_path = tmp_path / "notes_plugin_telemetry.jsonl"
    notes_plugin._action_token_weights = {}
    notes_plugin._note_selection_weights = {}
    return notes_plugin


@pytest_asyncio.fixture
async def pipeline() -> ContractPipeline:
    return app.state.container.pipeline


@pytest_asyncio.fixture
async def client(pipeline: ContractPipeline):
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
