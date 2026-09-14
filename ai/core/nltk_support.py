"""Resilient NLTK access for A.L.I.C.E.

NLTK ships code via pip but its corpora are a separate download. Guarding only
``ImportError`` is not enough: ``SentimentIntensityAnalyzer()`` and
``word_tokenize()`` raise ``LookupError`` at *call* time when the corpus is
absent, which turned a missing data file into a hard crash on a fresh install.

This module gives the rest of the runtime two things that never raise:

* :func:`tokenize` — NLTK's tokenizer when the corpus is present, a regex
  tokenizer otherwise.
* :func:`get_sentiment_analyzer` — a VADER analyzer, or ``None``.

:func:`ensure_corpora` downloads the corpora once when a network is available;
it is called by ``scripts/setup_nltk.py`` and is safe to call at runtime.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)

# Corpora the runtime actually uses, mapped to the resource path NLTK looks up.
REQUIRED_CORPORA: dict[str, str] = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "vader_lexicon": "sentiment/vader_lexicon",
    "stopwords": "corpora/stopwords",
}

_WORD_RE = re.compile(r"\w+(?:'\w+)?|[^\w\s]")

_lock = threading.Lock()
_tokenizer: Optional[Callable[[str], List[str]]] = None
_sentiment: Any = None
_sentiment_resolved = False


def _regex_tokenize(text: str) -> List[str]:
    """Tokenizer of last resort: words, contractions, and standalone punctuation."""
    return _WORD_RE.findall(text or "")


def _resolve_tokenizer() -> Callable[[str], List[str]]:
    try:
        from nltk import word_tokenize as _nltk_tokenize
    except Exception:
        logger.debug("NLTK not installed; using regex tokenizer")
        return _regex_tokenize

    # NLTK only touches the corpus on first call, so probe it once here rather
    # than letting a LookupError escape mid-turn.
    try:
        _nltk_tokenize("probe sentence.")
    except Exception as exc:  # LookupError, and anything else NLTK raises
        logger.warning("NLTK tokenizer data unavailable (%s); using regex tokenizer. Run scripts/setup_nltk.py", exc)
        return _regex_tokenize
    return _nltk_tokenize


def tokenize(text: str) -> List[str]:
    """Tokenize ``text``. Never raises, whatever NLTK data is installed."""
    global _tokenizer
    if _tokenizer is None:
        with _lock:
            if _tokenizer is None:
                _tokenizer = _resolve_tokenizer()
    try:
        return _tokenizer(text or "")
    except Exception:
        return _regex_tokenize(text)


def get_sentiment_analyzer() -> Any:
    """Return a VADER ``SentimentIntensityAnalyzer``, or ``None`` if unavailable."""
    global _sentiment, _sentiment_resolved
    if _sentiment_resolved:
        return _sentiment
    with _lock:
        if _sentiment_resolved:
            return _sentiment
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer

            _sentiment = SentimentIntensityAnalyzer()
        except Exception as exc:  # ImportError, LookupError, corpus corruption
            logger.warning("VADER sentiment unavailable (%s); sentiment will read neutral", exc)
            _sentiment = None
        _sentiment_resolved = True
    return _sentiment


def polarity_scores(text: str) -> dict:
    """VADER polarity for ``text``, or a neutral reading when VADER is absent."""
    analyzer = get_sentiment_analyzer()
    if analyzer is None:
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
    try:
        return analyzer.polarity_scores(text or "")
    except Exception:
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}


def missing_corpora() -> List[str]:
    """Names of the required corpora that are not installed."""
    try:
        import nltk
    except Exception:
        return sorted(REQUIRED_CORPORA)

    missing = []
    for name, resource in REQUIRED_CORPORA.items():
        try:
            nltk.data.find(resource)
        except LookupError:
            missing.append(name)
        except Exception:
            missing.append(name)
    return missing


def ensure_corpora(quiet: bool = True) -> List[str]:
    """Download any missing corpora. Returns the names still missing afterwards.

    Safe to call when offline — a failed download leaves the regex fallbacks in
    place rather than raising.
    """
    missing = missing_corpora()
    if not missing:
        return []

    try:
        import nltk
    except Exception:
        return missing

    for name in list(missing):
        try:
            if nltk.download(name, quiet=quiet):
                missing.remove(name)
        except Exception as exc:
            logger.debug("NLTK download of %s failed: %s", name, exc)

    # Re-resolve lazily so a successful download takes effect without a restart.
    global _tokenizer, _sentiment, _sentiment_resolved
    with _lock:
        _tokenizer = None
        _sentiment = None
        _sentiment_resolved = False
    return missing_corpora()
