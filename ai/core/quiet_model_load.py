"""Silence model-loading chatter without touching the process's streams.

sentence-transformers and its dependencies print progress bars and load reports
straight to stdout/stderr. The obvious way to hide that is
``contextlib.redirect_stdout``, but those swap ``sys.stdout`` *process-wide*:
when a model loads on a background thread, everything the main thread prints
during the load lands in the throwaway buffer and is lost. Alice warms her
classifier in the background precisely so startup is not blocked, so the
redirect approach silently ate the startup log.

Setting the libraries' own quiet switches achieves the same thing without any
global stream surgery, so it is safe on any thread.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Iterator

# Libraries whose import-time and load-time logging is noise for a local assistant.
_NOISY_LOGGERS = (
    "sentence_transformers",
    "sentence_transformers.SentenceTransformer",
    "transformers",
    "transformers.modeling_utils",
    "huggingface_hub",
    "paddlenlp",
    "paddlenlp.transformers",
    "torch",
    "filelock",
)

_QUIET_ENV = {
    "TQDM_DISABLE": "1",
    "TRANSFORMERS_VERBOSITY": "error",
    "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}


def apply_quiet_environment() -> None:
    """Set the quiet switches the model stack reads at import time.

    Only fills in values the caller has not already chosen, so an operator
    debugging a model load can still turn the noise back on.
    """
    for key, value in _QUIET_ENV.items():
        os.environ.setdefault(key, value)


# HTTP statuses that mean "asking again will get the same answer": the repo is
# gated, missing, or the caller is not authorised. Retrying these with backoff
# spends seconds to arrive at the same failure.
_PERMANENT_STATUSES = {400, 401, 403, 404, 405, 410, 422}

_PERMANENT_MARKERS = (
    "gatedrepo",
    "repositorynotfound",
    "entrynotfound",
    "unauthorized",
    "forbidden",
    "not a valid model identifier",
    "does not appear to have a file named",
)


def is_retryable_load_error(exc: BaseException) -> bool:
    """Whether retrying a model load could plausibly succeed.

    Offline or behind a proxy, a model download fails the same way every time.
    Retrying three times with exponential backoff turned a hopeless download into
    six seconds of sleeping before the assistant would talk to anyone.
    """
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is None:
        status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        if status in _PERMANENT_STATUSES:
            return False
        return status >= 500

    text = f"{type(exc).__name__} {exc}".lower()
    if any(marker in text for marker in _PERMANENT_MARKERS):
        return False
    # A bare status in the message, as huggingface_hub tends to render it.
    if any(f" {code} " in f" {text} " for code in map(str, _PERMANENT_STATUSES)):
        return False
    return True


@contextmanager
def quiet_model_load() -> Iterator[None]:
    """Raise the noisy loggers to ERROR for the duration of a model load."""
    apply_quiet_environment()
    previous = {}
    for name in _NOISY_LOGGERS:
        log = logging.getLogger(name)
        previous[name] = log.level
        log.setLevel(logging.ERROR)
    try:
        yield
    finally:
        for name, level in previous.items():
            logging.getLogger(name).setLevel(level)
