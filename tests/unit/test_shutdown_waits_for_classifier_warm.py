"""Shutdown waits for a classifier load that is still running, but not forever.

The warm is a daemon thread that loads torch. Exiting the interpreter while it is
inside torch aborts the process with SIGABRT ("terminate called without an active
exception"), which failed the startup-cost boot probe whenever the load outlasted
construction.
"""

import inspect
import threading
import time

from app.main import ALICE


def _alice_with_warm(target) -> ALICE:
    alice = ALICE.__new__(ALICE)
    alice._classifier_warm_thread = threading.Thread(target=target, daemon=True)
    alice._classifier_warm_thread.start()
    return alice


def test_shutdown_waits_for_a_load_that_is_about_to_finish():
    alice = _alice_with_warm(lambda: time.sleep(0.3))
    alice._wait_for_classifier_warm(timeout=5)
    assert not alice._classifier_warm_thread.is_alive()


def test_shutdown_does_not_hang_on_a_load_that_never_finishes():
    release = threading.Event()
    alice = _alice_with_warm(release.wait)
    started = time.perf_counter()
    alice._wait_for_classifier_warm(timeout=0.2)
    elapsed = time.perf_counter() - started
    release.set()
    assert elapsed < 5


def test_shutdown_waits_for_the_load():
    assert "self._wait_for_classifier_warm()" in inspect.getsource(ALICE.shutdown)
