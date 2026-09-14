#!/usr/bin/env python3
"""Download the NLTK corpora A.L.I.C.E uses.

NLTK ships its code via pip and its data separately, and the data lookups raise
at *call* time rather than import time. The runtime degrades gracefully without
them (regex tokenizer, neutral sentiment — see ai/core/nltk_support), but the
real tokenizer is better, so this fetches them once.

    python scripts/setup_nltk.py

Exits 0 when everything is present, 1 when something could not be fetched, so
CI can gate on it while a developer offline can ignore it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.core.nltk_support import REQUIRED_CORPORA, ensure_corpora, missing_corpora


def main() -> int:
    before = missing_corpora()
    if not before:
        print(f"All {len(REQUIRED_CORPORA)} NLTK corpora already present.")
        return 0

    print(f"Fetching {len(before)} missing NLTK corpora: {', '.join(before)}")
    still_missing = ensure_corpora(quiet=False)
    if still_missing:
        print(f"Could not fetch: {', '.join(still_missing)}", file=sys.stderr)
        print("A.L.I.C.E will fall back to a regex tokenizer and neutral sentiment.", file=sys.stderr)
        return 1

    print("All NLTK corpora installed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
