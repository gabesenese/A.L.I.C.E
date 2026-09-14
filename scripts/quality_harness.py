#!/usr/bin/env python3
"""Measure how well Alice actually answers, on your machine, against your model.

Nothing in the test suite can judge answer quality: tests run without Ollama, so
they can prove the plumbing works and nothing more. This runs a fixed set of
turns against a real ALICE and a real local model, checks each against objective
criteria (did she call the tool that had the answer, did she invent a number,
did she leak internal vocabulary, how long did it take), and prints a report.

    python scripts/quality_harness.py
    python scripts/quality_harness.py --only workspace_listing,multi_step
    python scripts/quality_harness.py --model llama3.1:8b --json run-a.json
    python scripts/quality_harness.py --compare run-a.json

The checks are deliberately mechanical — no model grades another model. They
catch the failures that matter most for a local assistant: fabricating facts a
tool could have supplied, not reaching for tools at all, reaching for them when
the turn was just conversation, and answering in the pipeline's vocabulary
instead of the user's.

Exit status is 0 when every scenario passes, 1 otherwise, so it can gate a
change you are unsure about.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SUITE = PROJECT_ROOT / "scenarios" / "quality" / "suite.json"

# Words that name Alice's internals rather than describing the user's situation.
INTERNAL_VOCABULARY = (
    "language model",
    "response path",
    "fast lane",
    "decision band",
    "contract pipeline",
    "tool_result",
    "plugin manager",
    "boundary_factory",
    "fallback policy",
    "verifier rejected",
)

# A reply that is only one of these is a non-answer wearing a sentence.
NON_ANSWER_PATTERNS = (
    r"^\s*i wasn'?t sure how to respond",
    r"^\s*i don'?t have a response right now",
    r"^\s*could you be more specific\s*\??\s*$",
    r"^\s*i didn'?t follow that",
)

# Weather numbers that did not come from the weather tool are invented.
WEATHER_FACT_PATTERN = re.compile(
    r"-?\d{1,3}\s*(?:°|degrees?\b|\bc\b|\bf\b)|humidity[^\n]{0,16}\d{1,3}\s*%",
    re.IGNORECASE,
)

WEATHER_TOOLS = {"get_current_weather", "weather", "weather:current", "weather:forecast"}


@dataclass
class TurnResult:
    id: str
    prompt: str
    why: str
    response: str = ""
    seconds: float = 0.0
    tools_used: List[str] = field(default_factory=list)
    route: str = ""
    intent: str = ""
    failures: List[str] = field(default_factory=list)
    error: str = ""

    @property
    def passed(self) -> bool:
        return not self.failures and not self.error

    @property
    def word_count(self) -> int:
        return len(self.response.split())


def _tools_from_metadata(meta: Dict[str, Any]) -> List[str]:
    """Collect every tool name the turn recorded, wherever it recorded it."""
    names: List[str] = []
    for key in ("tools_used", "tool_names"):
        names.extend(str(n) for n in (meta.get(key) or []))
    for step in meta.get("tool_steps") or []:
        if isinstance(step, dict) and step.get("tool"):
            names.append(str(step["tool"]))
    tool_result = meta.get("tool_result")
    if isinstance(tool_result, dict) and tool_result.get("tool_name"):
        names.append(str(tool_result["tool_name"]))
    # The plugin route records the intent it dispatched rather than a tool name.
    if str(meta.get("route") or "") in {"tool", "plugin"} and meta.get("intent"):
        names.append(str(meta["intent"]))
    seen, unique = set(), []
    for name in names:
        if name and name not in seen:
            seen.add(name)
            unique.append(name)
    return unique


def _check(result: TurnResult, checks: Dict[str, Any]) -> None:
    text = result.response
    lowered = text.lower()

    if checks.get("must_answer", True):
        if not text.strip():
            result.failures.append("empty response")
        elif any(re.search(p, lowered) for p in NON_ANSWER_PATTERNS):
            result.failures.append("non-answer: gave a shrug instead of an attempt")

    for term in INTERNAL_VOCABULARY:
        if term in lowered:
            result.failures.append(f"leaked internal vocabulary: {term!r}")

    if (limit := checks.get("max_words")) and result.word_count > int(limit):
        result.failures.append(f"too long: {result.word_count} words (max {limit})")

    if (floor := checks.get("min_words")) and result.word_count < int(floor):
        result.failures.append(f"too short: {result.word_count} words (min {floor})")

    for pattern in checks.get("must_match", []):
        if not re.search(pattern, lowered):
            result.failures.append(f"missing required pattern: {pattern}")

    for pattern in checks.get("must_not_match", []):
        if re.search(pattern, lowered):
            result.failures.append(f"matched forbidden pattern: {pattern}")

    if required_any := checks.get("must_include_any"):
        if not any(str(term).lower() in lowered for term in required_any):
            result.failures.append(f"none of the expected terms present: {required_any}")

    if expected_tools := checks.get("expects_any_tool"):
        if not any(t in result.tools_used for t in expected_tools):
            result.failures.append(f"called no tool from {expected_tools} (used: {result.tools_used or 'none'})")

    if (minimum := checks.get("min_tool_calls")) and len(result.tools_used) < int(minimum):
        result.failures.append(f"only {len(result.tools_used)} tool call(s), needed {minimum}")

    for forbidden in checks.get("must_not_have_used_tool", []):
        if forbidden in result.tools_used:
            result.failures.append(f"ran a tool it should not have: {forbidden}")

    if checks.get("must_not_have_used_any_tool") and result.tools_used:
        result.failures.append(f"reached for a tool on a conversational turn: {result.tools_used}")

    if checks.get("no_fabricated_weather"):
        cited = WEATHER_FACT_PATTERN.search(text)
        grounded = any(t in WEATHER_TOOLS or "weather" in t for t in result.tools_used)
        if cited and not grounded:
            result.failures.append(f"stated a weather figure ({cited.group(0).strip()!r}) without reading it")


def run_suite(
    suite: Dict[str, Any],
    *,
    model: Optional[str],
    only: Optional[List[str]],
    verbose: bool,
) -> List[TurnResult]:
    os.chdir(PROJECT_ROOT)
    os.environ.setdefault("ALICE_ENABLE_BACKGROUND_SERVICES", "0")

    from app.main import ALICE

    print("Booting A.L.I.C.E ...", flush=True)
    started = time.perf_counter()
    kwargs: Dict[str, Any] = {"user_name": "Tester", "debug": False}
    if model:
        kwargs["llm_model"] = model
    alice = ALICE(**kwargs)
    print(f"Ready in {time.perf_counter() - started:.1f}s\n", flush=True)

    results: List[TurnResult] = []
    try:
        for scenario in suite.get("scenarios", []):
            scenario_id = str(scenario.get("id") or "?")
            if only and scenario_id not in only:
                continue

            result = TurnResult(
                id=scenario_id,
                prompt=str(scenario.get("prompt") or ""),
                why=str(scenario.get("why") or ""),
            )
            alice.last_turn_metadata = {}
            turn_started = time.perf_counter()
            try:
                result.response = str(alice.process_input(result.prompt) or "")
            except Exception as exc:
                result.error = f"{type(exc).__name__}: {exc}"
            result.seconds = time.perf_counter() - turn_started

            meta = dict(getattr(alice, "last_turn_metadata", {}) or {})
            result.tools_used = _tools_from_metadata(meta)
            result.route = str(meta.get("route") or "")
            result.intent = str(meta.get("intent") or "")

            if not result.error:
                _check(result, dict(scenario.get("checks") or {}))

            results.append(result)
            mark = "PASS" if result.passed else "FAIL"
            print(f"[{mark}] {scenario_id}  ({result.seconds:.1f}s, route={result.route or '-'})")
            if verbose or not result.passed:
                print(f"       > {result.prompt}")
                print(f"       {result.response.strip()[:400] or '(no response)'}")
                if result.tools_used:
                    print(f"       tools: {', '.join(result.tools_used)}")
                for failure in result.failures:
                    print(f"       ! {failure}")
                if result.error:
                    print(f"       ! raised {result.error}")
                print()
    finally:
        try:
            alice.shutdown()
        except Exception:
            pass

    return results


def summarise(results: List[TurnResult]) -> Dict[str, Any]:
    if not results:
        return {}
    latencies = [r.seconds for r in results]
    with_tools = [r for r in results if r.tools_used]
    return {
        "scenarios": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "tool_use_rate": round(len(with_tools) / len(results), 3),
        "median_seconds": round(statistics.median(latencies), 2),
        "slowest_seconds": round(max(latencies), 2),
        "slowest_scenario": max(results, key=lambda r: r.seconds).id,
    }


def print_report(results: List[TurnResult], summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 68)
    print("SUMMARY")
    print("=" * 68)
    for key, value in summary.items():
        print(f"  {key.replace('_', ' '):<20} {value}")

    failed = [r for r in results if not r.passed]
    if failed:
        print("\nWhat to look at first:")
        for result in failed:
            print(f"\n  {result.id} — {result.why}")
            for failure in result.failures or ([result.error] if result.error else []):
                print(f"      ! {failure}")
    else:
        print("\n  Every scenario passed.")


def compare(previous_path: Path, results: List[TurnResult]) -> None:
    previous = {r["id"]: r for r in json.loads(previous_path.read_text())["results"]}
    print("\n" + "=" * 68)
    print(f"COMPARED WITH {previous_path.name}")
    print("=" * 68)
    for result in results:
        before = previous.get(result.id)
        if before is None:
            print(f"  {result.id:<22} NEW")
            continue
        was, now = bool(before["passed"]), result.passed
        delta = result.seconds - float(before.get("seconds") or 0.0)
        if was != now:
            print(f"  {result.id:<22} {'FIXED' if now else 'REGRESSED'}   ({delta:+.1f}s)")
        else:
            print(f"  {result.id:<22} {'pass' if now else 'fail'}      ({delta:+.1f}s)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE, help="Scenario file to run.")
    parser.add_argument("--model", help="Ollama model to use, e.g. llama3.1:8b.")
    parser.add_argument("--only", help="Comma-separated scenario ids to run.")
    parser.add_argument("--json", type=Path, help="Write the full run here for later comparison.")
    parser.add_argument("--compare", type=Path, help="Compare this run against a previous --json file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print every response, not just failures.")
    args = parser.parse_args()

    if not args.suite.exists():
        print(f"No such suite: {args.suite}", file=sys.stderr)
        return 2

    suite = json.loads(args.suite.read_text())
    only = [s.strip() for s in args.only.split(",")] if args.only else None

    results = run_suite(suite, model=args.model, only=only, verbose=args.verbose)
    if not results:
        print("No scenarios ran.", file=sys.stderr)
        return 2

    summary = summarise(results)
    print_report(results, summary)

    if args.json:
        payload = {
            "suite": suite.get("name", args.suite.stem),
            "model": args.model or "default",
            "summary": summary,
            "results": [{**asdict(r), "passed": r.passed, "word_count": r.word_count} for r in results],
        }
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.json}")

    if args.compare:
        if args.compare.exists():
            compare(args.compare, results)
        else:
            print(f"\nNothing to compare against: {args.compare} does not exist", file=sys.stderr)

    return 0 if summary.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
