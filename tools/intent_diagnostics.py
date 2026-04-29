"""
tools/intent_diagnostics.py
────────────────────────────
Offline diagnostic script for the A.L.I.C.E intent-routing pipeline.

Reads ``memory/auto_generated_corrections.jsonl`` (the file written by
NLPErrorLogger whenever active-learning overrides an intent) and produces:

1. A per-intent precision / recall summary (stdout or --output <file>).
2. A confusion-matrix CSV ready for import into a spreadsheet.
3. A BayesianIntentRouter warm-up call that seeds the calibrator with the
   correction history so the router is no longer cold-started on next run.

Usage
-----
    python tools/intent_diagnostics.py
    python tools/intent_diagnostics.py --corrections memory/auto_generated_corrections.jsonl
    python tools/intent_diagnostics.py --output reports/intent_diagnostics.md --confusion-csv reports/confusion.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List


# ── helpers ──────────────────────────────────────────────────────────────────


def _load_corrections(path: Path) -> List[dict]:
    """Load JSONL correction records. Each line is one override event."""
    records: List[dict] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records


def _build_confusion(records: List[dict]) -> Dict[str, Dict[str, int]]:
    """
    Build confusion matrix: predicted[original_intent][corrected_intent] = count.

    Keys: original_intent (what NLP said) → corrected_intent (what was right).
    """
    matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in records:
        predicted = r.get("original_intent") or r.get("intent") or "unknown"
        actual = r.get("corrected_intent") or r.get("correction") or "unknown"
        matrix[predicted][actual] += 1
    return {k: dict(v) for k, v in matrix.items()}


def _precision_recall(matrix: Dict[str, Dict[str, int]]) -> Dict[str, dict]:
    """
    Compute per-intent precision and recall from the confusion matrix.

    precision(intent) = TP / (TP + FP)  — of all times NLP predicted this
                                           intent, how often was it correct?
    recall(intent)    = TP / (TP + FN)  — of all times the true intent was
                                           this, how often did NLP catch it?
    """
    # Count TP, FP, FN per intent
    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()

    for predicted, actual_counts in matrix.items():
        for actual, count in actual_counts.items():
            if predicted == actual:
                tp[predicted] += count
            else:
                fp[predicted] += count  # NLP said predicted, real was actual
                fn[actual] += count  # NLP missed real intent=actual

    all_intents = set(tp.keys()) | set(fp.keys()) | set(fn.keys())
    stats: Dict[str, dict] = {}
    for intent in sorted(all_intents):
        _tp = tp[intent]
        _fp = fp[intent]
        _fn = fn[intent]
        prec = _tp / (_tp + _fp) if (_tp + _fp) > 0 else 0.0
        rec = _tp / (_tp + _fn) if (_tp + _fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        stats[intent] = {
            "tp": _tp,
            "fp": _fp,
            "fn": _fn,
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1": round(f1, 3),
            "support": _tp + _fn,
        }
    return stats


def _write_confusion_csv(
    matrix: Dict[str, Dict[str, int]],
    out_path: Path,
) -> None:
    all_intents = sorted(set(matrix.keys()) | {a for v in matrix.values() for a in v})
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["predicted \\ actual"] + all_intents)
        for predicted in all_intents:
            row = [predicted]
            for actual in all_intents:
                row.append(matrix.get(predicted, {}).get(actual, 0))
            writer.writerow(row)


def _write_markdown_report(
    stats: Dict[str, dict],
    total_corrections: int,
    out: "TextIO",
) -> None:
    out.write("# A.L.I.C.E Intent Diagnostics\n\n")
    out.write(f"Total correction events: **{total_corrections}**\n\n")
    out.write(
        "| Intent | Precision | Recall | F1 | Support | TP | FP | FN |\n"
        "|--------|-----------|--------|----|---------|----|----|----|" + "\n"
    )
    for intent, s in stats.items():
        out.write(
            f"| `{intent}` | {s['precision']:.3f} | {s['recall']:.3f} | "
            f"{s['f1']:.3f} | {s['support']} | {s['tp']} | {s['fp']} | {s['fn']} |\n"
        )
    out.write("\n")


def _warm_up_router(records: List[dict]) -> None:
    """
    Feed historical corrections into BayesianIntentRouter's calibrator so
    the confidence curve is no longer flat on next run.
    """
    try:
        from ai.core.intent_classifier import get_bayesian_router

        router = get_bayesian_router()
        warmed = 0
        for r in records:
            original = r.get("original_intent") or r.get("intent")
            corrected = r.get("corrected_intent") or r.get("correction")
            conf = float(r.get("original_confidence") or r.get("confidence") or 0.5)
            if original and corrected:
                was_correct = original == corrected
                router.record_outcome(
                    original, was_correct=was_correct, confidence=conf
                )
                warmed += 1
        print(
            f"[warm-up] Seeded BayesianIntentRouter with {warmed} historical records."
        )
    except Exception as exc:
        print(f"[warm-up] Router warm-up skipped: {exc}")


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate intent precision/recall diagnostics from correction logs."
    )
    parser.add_argument(
        "--corrections",
        default="memory/auto_generated_corrections.jsonl",
        help="Path to the corrections JSONL file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write Markdown report to this path (default: stdout).",
    )
    parser.add_argument(
        "--confusion-csv",
        dest="confusion_csv",
        default=None,
        help="Write confusion-matrix CSV to this path.",
    )
    parser.add_argument(
        "--warm-up",
        dest="warm_up",
        action="store_true",
        default=False,
        help="Seed BayesianIntentRouter calibrator with correction history.",
    )
    args = parser.parse_args()

    corrections_path = Path(args.corrections)
    records = _load_corrections(corrections_path)
    if not records:
        print(f"No records found in {corrections_path}. Nothing to analyse.")
        sys.exit(0)

    matrix = _build_confusion(records)
    stats = _precision_recall(matrix)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            _write_markdown_report(stats, len(records), fh)
        print(f"Report written to {out_path}")
    else:
        import io

        buf = io.StringIO()
        _write_markdown_report(stats, len(records), buf)
        print(buf.getvalue())

    if args.confusion_csv:
        csv_path = Path(args.confusion_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        _write_confusion_csv(matrix, csv_path)
        print(f"Confusion matrix written to {csv_path}")

    if args.warm_up:
        _warm_up_router(records)


if __name__ == "__main__":
    main()
