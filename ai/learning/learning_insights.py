"""
Learning Insights — reads quality_issues.jsonl and generates actionable reports.

Run from the command line:
    python -m ai.learning.learning_insights

Or call from within ALICE via the /analyze-learning command.
"""

import json
import re
import logging
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import Dict, List, Any

from ai.learning.response_quality_checker import (
    ISSUE_DIRECTNESS,
    ISSUE_REPETITION,
    ISSUE_VOCAB_GAP,
    ISSUE_UNNECESSARY_PLUGIN,
)

logger = logging.getLogger(__name__)

QUALITY_LOG = Path("data/realtime_learning/quality_issues.jsonl")


class LearningInsights:
    """
    Reads accumulated quality issues and produces:
    - A plain-text report with concrete fix suggestions
    - Structured data for programmatic consumption

    The report answers: "What should the dev change to prevent these errors?"
    """

    def __init__(self, lookback_days: int = 30):
        self.lookback_days = lookback_days
        self._issues: List[Dict[str, Any]] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self) -> "LearningInsights":
        """Load issues from disk (call before generate_report)."""
        if not QUALITY_LOG.exists():
            return self
        cutoff = datetime.now() - timedelta(days=self.lookback_days)
        loaded = []
        try:
            with open(QUALITY_LOG, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        ts = datetime.fromisoformat(
                            entry.get("timestamp", "2000-01-01")
                        )
                        if ts >= cutoff:
                            loaded.append(entry)
                    except (json.JSONDecodeError, ValueError):
                        pass
        except Exception as e:
            logger.warning("[LearningInsights] Failed to load: %s", e)
        self._issues = loaded
        return self

    def generate_report(self) -> str:
        """Return a human-readable improvement report with specific fix suggestions."""
        if not self._issues:
            return (
                "No quality issues recorded yet.\n"
                "Issues are accumulated as ALICE has conversations.\n"
                "Re-run after more interactions."
            )

        lines = [
            f"{'='*70}",
            "  A.L.I.C.E  ·  Learning Insights Report",
            f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"  Issues analysed: {len(self._issues)} (last {self.lookback_days} days)",
            f"{'='*70}",
            "",
        ]

        # ── Section 1: Vocab gaps ──────────────────────────────────────────
        vocab_gaps = self._vocab_gap_summary()
        if vocab_gaps:
            lines.append(
                "── VOCABULARY GAPS ──────────────────────────────────────────────"
            )
            lines.append(
                "Words that appeared in domain queries but aren't in keyword lists."
            )
            lines.append(
                "These cause the fast-path to miss queries and call plugins unnecessarily."
            )
            lines.append("")
            for domain, words in sorted(
                vocab_gaps.items(), key=lambda x: -sum(x[1].values())
            ):
                top = sorted(words.items(), key=lambda x: -x[1])[:10]
                lines.append(f"  [{domain}]")
                for word, count in top:
                    lines.append(f'    • "{word}"  — seen {count}x')
                lines.append(f"  → Suggested fix: add to {domain} keyword list:")
                kw_str = ", ".join(f"'{w}'" for w, _ in top[:6])
                lines.append(f"    {kw_str}")
                lines.append("")

        # ── Section 2: Directness failures ────────────────────────────────
        direct_fails = [i for i in self._issues if i["issue_type"] == ISSUE_DIRECTNESS]
        if direct_fails:
            lines.append(
                "── DIRECTNESS FAILURES ──────────────────────────────────────────"
            )
            lines.append("Yes/No questions that received an answer without Yes/No.")
            lines.append("")
            by_input = Counter(i["user_input"].lower()[:60] for i in direct_fails)
            for inp, count in by_input.most_common(5):
                lines.append(f'  {count}x  "{inp}"')
            lines.append("")
            lines.append(
                "  → Fix: Check _handle_weather_followup / _format_response templates."
            )
            lines.append(
                "         Responses to 'should I / do I need' must start with Yes/No."
            )
            lines.append("")

        # ── Section 3: Info repetition ────────────────────────────────────
        repeats = [i for i in self._issues if i["issue_type"] == ISSUE_REPETITION]
        if repeats:
            lines.append(
                "── INFO REPETITION ──────────────────────────────────────────────"
            )
            lines.append(
                "Responses that restated the same facts as the immediately previous turn."
            )
            lines.append("")
            by_domain = Counter(i["domain"] for i in repeats)
            for domain, count in by_domain.most_common():
                lines.append(f"  {domain}: {count} occurrences")
                sample = next(
                    (
                        i["detail"].get("repeated_phrases", [])
                        for i in repeats
                        if i["domain"] == domain
                    ),
                    [],
                )
                if sample:
                    lines.append(f'    e.g. repeated phrase: "{sample[0]}"')
            lines.append("")
            lines.append(
                "  → Fix: Use is_follow_up flag to shorten consecutive same-domain responses."
            )
            lines.append("")

        # ── Section 4: Unnecessary plugin calls ───────────────────────────
        plugin_calls = [
            i for i in self._issues if i["issue_type"] == ISSUE_UNNECESSARY_PLUGIN
        ]
        if plugin_calls:
            lines.append(
                "── UNNECESSARY PLUGIN CALLS ─────────────────────────────────────"
            )
            lines.append("Plugin called even though stored data was already available.")
            lines.append("")
            by_plugin = Counter(i["detail"].get("plugin", "?") for i in plugin_calls)
            for plugin, count in by_plugin.most_common():
                lines.append(f"  {plugin}: {count} unnecessary calls")
                # Collect the trigger words that caused the miss
                all_words: List[str] = []
                for issue in plugin_calls:
                    if issue["detail"].get("plugin") == plugin:
                        all_words += issue["detail"].get("user_words", [])
                top_words = Counter(all_words).most_common(6)
                if top_words:
                    words_str = ", ".join(f'"{w}"' for w, _ in top_words)
                    lines.append(f"    Input words not in fast-path: {words_str}")
            lines.append("")
            lines.append(
                "  → Fix: Add those words to weather_followup_indicators / _handle_weather_followup."
            )
            lines.append("")

        # ── Summary ───────────────────────────────────────────────────────
        by_type = Counter(i["issue_type"] for i in self._issues)
        lines.append(
            "── SUMMARY ──────────────────────────────────────────────────────"
        )
        for issue_type, count in by_type.most_common():
            lines.append(f"  {issue_type:<40} {count:>4}")
        lines.append("")
        lines.append(f"{'='*70}")

        return "\n".join(lines)

    def get_vocab_gap_suggestions(self) -> Dict[str, List[str]]:
        """Return {domain: [missing_words]} sorted by frequency, for programmatic use."""
        summary = self._vocab_gap_summary()
        return {
            domain: [w for w, _ in sorted(words.items(), key=lambda x: -x[1])[:10]]
            for domain, words in summary.items()
        }

    def get_issue_counts(self) -> Dict[str, int]:
        return dict(Counter(i["issue_type"] for i in self._issues))

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _vocab_gap_summary(self) -> Dict[str, Counter]:
        """Return {domain: Counter(word → count)} for vocab gap issues."""
        result: Dict[str, Counter] = defaultdict(Counter)
        for issue in self._issues:
            if issue["issue_type"] != ISSUE_VOCAB_GAP:
                continue
            domain = issue["domain"]
            gap_words = issue.get("detail", {}).get("gap_words", [])
            for word in gap_words:
                result[domain][word] += 1
        return dict(result)


# ── CLI entry point ───────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ALICE Learning Insights Report")
    parser.add_argument("--days", type=int, default=30, help="Look-back window in days")
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear the quality issues log after reporting",
    )
    args = parser.parse_args()

    insights = LearningInsights(lookback_days=args.days).load()
    print(insights.generate_report())

    if args.clear and QUALITY_LOG.exists():
        QUALITY_LOG.unlink()
        print("\nQuality issues log cleared.")


if __name__ == "__main__":
    main()
