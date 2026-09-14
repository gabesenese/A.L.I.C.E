"""Startup must not spend the user's time on work the first turn may never need.

Booting ALICE is expensive, so this does it once in a subprocess and reports
everything the assertions below need as one JSON blob.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BOOT_BUDGET_SECONDS = 300

# Construction must stay well under this. It is a ceiling on "did we reintroduce
# a blocking model load or a network retry storm", not a performance target.
INIT_BUDGET_SECONDS = 60

PROBE = """
import json, sys, time

started = time.perf_counter()
from app.main import ALICE
import_seconds = time.perf_counter() - started

started = time.perf_counter()
alice = ALICE(user_name="Tester", debug=False)
init_seconds = time.perf_counter() - started

print("MARKER-AFTER-INIT")

warm = getattr(alice, "_classifier_warm_thread", None)
report = {
    "import_seconds": import_seconds,
    "init_seconds": init_seconds,
    "warm_thread_present": warm is not None,
    "warm_thread_is_daemon": bool(warm is not None and warm.daemon),
    "subsystems": {
        name: getattr(alice, name, None) is not None
        for name in ("nlp", "memory", "plugins", "llm", "contract_pipeline")
    },
    "stdout_is_a_real_stream": sys.stdout is sys.__stdout__,
}
alice.shutdown()
print("MARKER-AFTER-SHUTDOWN")
print("REPORT " + json.dumps(report))
"""


@pytest.fixture(scope="module")
def boot():
    env = dict(os.environ)
    env["ALICE_ENABLE_BACKGROUND_SERVICES"] = "0"
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(PROBE)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=BOOT_BUDGET_SECONDS,
        env=env,
    )
    assert result.returncode == 0, f"boot failed\nstderr tail:\n{result.stderr[-4000:]}"
    line = next((ln for ln in result.stdout.splitlines() if ln.startswith("REPORT ")), None)
    assert line, f"boot produced no report\nstdout tail:\n{result.stdout[-3000:]}"
    return {"report": json.loads(line[len("REPORT ") :]), "stdout": result.stdout}


def test_constructing_alice_does_not_block_on_the_semantic_model(boot):
    """The classifier is lazy by design, and startup used to defeat that with a
    'pre-warm' call. Loading it imports torch and reads a sentence-transformers
    model; uncached, it retried with backoff — a minute or more of blocking
    before the prompt appeared, on a machine that is offline for the very reason
    someone runs a local assistant."""
    assert boot["report"]["init_seconds"] < INIT_BUDGET_SECONDS


def test_the_classifier_warm_runs_on_a_background_daemon_thread(boot):
    assert boot["report"]["warm_thread_present"] is True
    assert boot["report"]["warm_thread_is_daemon"] is True


def test_startup_output_is_not_swallowed_by_the_background_warm(boot):
    """The warm thread loads a model, and the loader used to swap sys.stdout
    process-wide while doing so, so whatever the main thread printed during the
    load was lost — including the rest of the startup log."""
    assert "MARKER-AFTER-INIT" in boot["stdout"]
    assert "MARKER-AFTER-SHUTDOWN" in boot["stdout"]
    assert boot["report"]["stdout_is_a_real_stream"] is True


@pytest.mark.parametrize("subsystem", ["nlp", "memory", "plugins", "llm", "contract_pipeline"])
def test_the_core_subsystems_are_present_after_construction(boot, subsystem):
    """Moving work off the critical path must not leave anything unbuilt."""
    assert boot["report"]["subsystems"][subsystem] is True


def test_shutdown_completes_cleanly(boot):
    assert "MARKER-AFTER-SHUTDOWN" in boot["stdout"]
