"""Loading a model must not silence the rest of the process, or sleep pointlessly."""

import io
import logging
import sys

import pytest

from ai.core.quiet_model_load import (
    apply_quiet_environment,
    is_retryable_load_error,
    quiet_model_load,
)


def test_quiet_model_load_leaves_the_process_streams_alone():
    """Regression: the old approach used contextlib.redirect_stdout, which swaps
    sys.stdout process-wide. Alice warms her classifier on a background thread so
    startup is not blocked, and everything the main thread printed during the
    load went into the throwaway buffer — the startup log simply vanished."""
    before_out, before_err = sys.stdout, sys.stderr
    with quiet_model_load():
        assert sys.stdout is before_out
        assert sys.stderr is before_err
    assert (sys.stdout, sys.stderr) == (before_out, before_err)


def test_output_from_another_thread_survives_a_model_load():
    import threading

    captured = io.StringIO()
    original = sys.stdout
    sys.stdout = captured
    try:
        done = threading.Event()

        def loader():
            with quiet_model_load():
                done.wait(timeout=2)

        worker = threading.Thread(target=loader)
        worker.start()
        print("startup line that must not be swallowed")
        done.set()
        worker.join(timeout=5)
    finally:
        sys.stdout = original

    assert "startup line that must not be swallowed" in captured.getvalue()


def test_noisy_logger_levels_are_restored():
    log = logging.getLogger("transformers")
    log.setLevel(logging.DEBUG)
    with quiet_model_load():
        assert log.level == logging.ERROR
    assert log.level == logging.DEBUG


def test_quiet_environment_does_not_override_an_explicit_choice(monkeypatch):
    monkeypatch.setenv("TRANSFORMERS_VERBOSITY", "debug")
    apply_quiet_environment()
    import os

    assert os.environ["TRANSFORMERS_VERBOSITY"] == "debug"
    assert os.environ["TQDM_DISABLE"] == "1"


# -- retryability ------------------------------------------------------------


class _Response:
    def __init__(self, status_code):
        self.status_code = status_code


class _HttpError(Exception):
    def __init__(self, status_code, message=""):
        super().__init__(message or f"{status_code} error")
        self.response = _Response(status_code)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 410, 422])
def test_a_permanent_http_failure_is_not_retried(status):
    """Offline or behind a proxy the download fails identically every time.
    Three attempts with backoff spent six seconds reaching the same answer
    before Alice would speak to anyone."""
    assert is_retryable_load_error(_HttpError(status)) is False


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_a_server_side_failure_is_retried(status):
    assert is_retryable_load_error(_HttpError(status)) is True


@pytest.mark.parametrize(
    "exc",
    [TimeoutError("timed out"), ConnectionError("connection reset"), OSError("temporary failure")],
)
def test_a_transient_network_failure_is_retried(exc):
    assert is_retryable_load_error(exc) is True


@pytest.mark.parametrize(
    "message",
    [
        "403 Forbidden",
        "GatedRepoError: access to model is restricted",
        "RepositoryNotFoundError: not a valid model identifier",
        "401 Client Error: Unauthorized",
    ],
)
def test_a_permanent_failure_is_recognised_from_its_message(message):
    assert is_retryable_load_error(RuntimeError(message)) is False
