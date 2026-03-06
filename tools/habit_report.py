"""
tools/habit_report.py
──────────────────────
Offline script that reads the HabitMiner's persistence file
(``memory/habits.jsonl``) and emits a Markdown dashboard to
``reports/habits.md`` summarising the top automation macros that
A.L.I.C.E has learned from observed user behaviour.

Also optionally prints anomaly and self-debug summaries when those
log files are present.

Usage
-----
    python tools/habit_report.py
    python tools/habit_report.py --habits memory/habits.jsonl --output reports/habits.md
    python tools/habit_report.py --anomalies --debug-log
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional


# ── loaders ──────────────────────────────────────────────────────────────────

def _load_habits(path: Path) -> List[dict]:
    habits: List[dict] = []
    if not path.exists():
        return habits
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            habits.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return habits


def _load_jsonl(path: Path) -> List[dict]:
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


# ── formatters ───────────────────────────────────────────────────────────────

def _format_habits(habits: List[dict]) -> str:
    if not habits:
        return "_No habits mined yet._\n"

    # Sort by confidence descending
    habits = sorted(habits, key=lambda h: h.get("confidence", 0), reverse=True)

    lines = ["| # | Macro Name | Sequence | Confidence | Trigger Length |",
             "|---|------------|----------|------------|----------------|"]
    for i, h in enumerate(habits[:30], 1):
        name = h.get("name", "unnamed")
        seq  = h.get("sequence", [])
        # sequence items can be strings or dicts
        seq_str = " → ".join(
            f"{s['intent']}:{s['plugin']}" if isinstance(s, dict) else str(s)
            for s in seq
        )
        conf   = h.get("confidence", 0)
        trig   = h.get("trigger_length", "?")
        lines.append(f"| {i} | `{name}` | {seq_str} | {conf:.1f} | {trig} |")
    return "\n".join(lines) + "\n"


def _format_anomalies(records: List[dict]) -> str:
    if not records:
        return "_No anomaly records._\n"

    lines = ["| # | Metric | Value | Threshold | Timestamp |",
             "|---|--------|-------|-----------|-----------|"]
    for i, r in enumerate(records[-20:], 1):
        metric = r.get("metric", "?")
        val    = r.get("value", "?")
        thr    = r.get("threshold", "?")
        ts     = r.get("timestamp", "?")[:19]
        lines.append(f"| {i} | `{metric}` | {val} | {thr} | {ts} |")
    return "\n".join(lines) + "\n"


def _format_debug_fixes(records: List[dict]) -> str:
    if not records:
        return "_No self-debug fix records._\n"

    lines = ["| # | Strategy | Failure Type | Target | Confidence | Timestamp |",
             "|---|----------|--------------|--------|------------|-----------|"]
    for i, r in enumerate(records[-20:], 1):
        strat = r.get("strategy", "?")
        ftype = r.get("failure_type", "?")
        tgt   = r.get("target", "?")
        conf  = r.get("confidence", 0)
        ts    = r.get("timestamp", "?")[:19]
        lines.append(f"| {i} | `{strat}` | `{ftype}` | `{tgt}` | {conf:.2f} | {ts} |")
    return "\n".join(lines) + "\n"


# ── top-level report builder ──────────────────────────────────────────────────

def build_report(
    habits_path: Path,
    anomalies_path: Optional[Path],
    debug_fixes_path: Optional[Path],
) -> str:
    habits   = _load_habits(habits_path)
    sections = [
        "# A.L.I.C.E Habit & Debug Dashboard\n",
        f"Habit macros learned: **{len(habits)}** | source: `{habits_path}`\n",
        "\n## Top Automation Macros\n",
        _format_habits(habits),
    ]

    if anomalies_path:
        anomalies = _load_jsonl(anomalies_path)
        sections += [
            f"\n## Response Anomalies (last 20 of {len(anomalies)})\n",
            _format_anomalies(anomalies),
        ]

    if debug_fixes_path:
        fixes = _load_jsonl(debug_fixes_path)
        sections += [
            f"\n## SelfDebugger Fix Actions (last 20 of {len(fixes)})\n",
            _format_debug_fixes(fixes),
        ]

    return "".join(sections)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a Markdown dashboard from A.L.I.C.E habit/debug logs."
    )
    parser.add_argument(
        "--habits",
        default="memory/habits.jsonl",
        help="Path to HabitMiner persistence file.",
    )
    parser.add_argument(
        "--output",
        default="reports/habits.md",
        help="Write Markdown report to this path.",
    )
    parser.add_argument(
        "--anomalies",
        dest="anomalies",
        action="store_true",
        default=False,
        help="Include anomaly log (memory/anomalies.jsonl) in report.",
    )
    parser.add_argument(
        "--anomalies-path",
        default="memory/anomalies.jsonl",
        help="Path to anomaly log (used when --anomalies is set).",
    )
    parser.add_argument(
        "--debug-log",
        dest="debug_log",
        action="store_true",
        default=False,
        help="Include SelfDebugger fix log (memory/self_debug_fixes.jsonl).",
    )
    parser.add_argument(
        "--debug-path",
        default="memory/self_debug_fixes.jsonl",
        help="Path to SelfDebugger fix log (used when --debug-log is set).",
    )
    args = parser.parse_args()

    habits_path       = Path(args.habits)
    anomalies_path    = Path(args.anomalies_path) if args.anomalies else None
    debug_fixes_path  = Path(args.debug_path) if args.debug_log else None

    report = build_report(habits_path, anomalies_path, debug_fixes_path)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"Report written to {out_path}")

    # Also print summary to stdout
    habits = _load_habits(habits_path)
    print(f"Habits loaded: {len(habits)}")
    if habits:
        top = sorted(habits, key=lambda h: h.get("confidence", 0), reverse=True)[:5]
        print("Top 5 macros:")
        for h in top:
            seq = h.get("sequence", [])
            seq_str = " → ".join(
                f"{s['intent']}:{s['plugin']}" if isinstance(s, dict) else str(s)
                for s in seq
            )
            print(f"  [{h.get('confidence', 0):.1f}] {h.get('name', '?')}: {seq_str}")


if __name__ == "__main__":
    main()
