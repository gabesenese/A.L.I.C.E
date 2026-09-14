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

# -- conversational feel -----------------------------------------------------
#
# The complaint these measure is "it feels like I'm talking to a terminal". That
# is not one defect; it is an accumulation of registers. Each pattern below is a
# way text announces it was assembled rather than said.

# Status-line and report vocabulary. A person does not say "acknowledged".
TERMINAL_REGISTER = (
    r"\backnowledged\b",
    r"\bplease provide\b",
    r"\bplease specify\b",
    r"\bi am unable to\b",
    r"\bunable to comply\b",
    r"\binvalid (?:input|request|command)\b",
    r"\b(?:operation|request|task) (?:completed|failed) successfully\b",
    r"^\s*\[(?:ok|error|warning|info|done)\]",
    r"^\s*(?:status|result|output|summary)\s*:",
    r"={4,}|-{4,}",
    r"\bas an ai\b",
)

# A reply that only hedges has taken no position at all.
HEDGE_PATTERNS = (
    r"it (?:really )?depends",
    r"there (?:is|are) no (?:one|single) (?:right )?answer",
    r"both (?:have|has) (?:their )?(?:pros and cons|advantages)",
    r"that'?s a (?:great|good|complex|nuanced) question",
    r"it'?s hard to say",
)

# Every turn ending in an offer to help is a script, not a conversation.
TRAILING_OFFER = re.compile(
    r"(?:let me know|would you like|shall i|do you want me to|"
    r"i can (?:help|assist)|feel free to)\b[^.!?]*[.!?]?\s*$",
    re.IGNORECASE,
)

# Three or more numbered or bulleted items in a row is a document, not an answer.
LIST_SHAPE = re.compile(r"(?:^|\n)\s*(?:[-*•]|\d+[.)])\s+\S", re.MULTILINE)

# A syllabus: "1) Foundations ... 2) Practical ... 3) Advanced".
SYLLABUS = re.compile(r"\b(?:phase|step|week|module|stage)\s*\d|\b\d\)\s*\w+:", re.IGNORECASE)

# Openings that restate the question instead of answering it.
RESTATEMENT = re.compile(
    r"^\s*(?:you(?:'re| are) asking|so,? you want|to answer your question|"
    r"regarding your question|in response to your)",
    re.IGNORECASE,
)

# Losing the thread: answering a follow-up by asking what it refers to.
LOST_THREAD = re.compile(
    r"(?:what|which)\s+(?:do you mean|are you referring to|is \"?it\"?)|"
    r"could you (?:clarify|specify|tell me more about) (?:what|which)|"
    r"i(?:'m| am) not sure what (?:you mean|\"?it\"? refers)",
    re.IGNORECASE,
)


def _matches_any(patterns, text: str) -> List[str]:
    return [p for p in patterns if re.search(p, text, re.IGNORECASE | re.MULTILINE)]


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

    # -- conversational feel --------------------------------------------------

    if checks.get("no_terminal_register"):
        for pattern in _matches_any(TERMINAL_REGISTER, text):
            result.failures.append(f"status-report register, not speech: {pattern}")

    if checks.get("no_hedge_only") and _matches_any(HEDGE_PATTERNS, text):
        # Hedging is only a failure when it is the whole reply. A hedge followed
        # by an actual position is a person being careful.
        without_hedge = text
        for pattern in HEDGE_PATTERNS:
            without_hedge = re.sub(pattern, "", without_hedge, flags=re.IGNORECASE)
        if len(without_hedge.split()) < max(12, result.word_count // 3):
            result.failures.append("hedged without taking a position")

    if checks.get("no_bullet_dump") and len(LIST_SHAPE.findall(text)) >= 3:
        result.failures.append("answered a human moment with a bulleted list")

    if checks.get("no_numbered_syllabus") and SYLLABUS.search(text):
        result.failures.append("delivered a syllabus rather than an explanation")

    if checks.get("no_lost_thread") and LOST_THREAD.search(text):
        result.failures.append("lost the thread: asked what the follow-up referred to")

    if checks.get("no_restatement") and RESTATEMENT.search(text):
        result.failures.append("opened by restating the question")


# Openings that only ever appear in a prompt Alice wrote to herself. If one of
# these turns up in conversation_history, the transcript she replays as "what we
# were talking about" contains machinery, and she will imitate its register.
MACHINE_PROMPT_MARKERS = (
    # An imperative aimed at a *thing* rather than at Alice. "Summarise what we
    # decided" is a person; "Summarise the conversation below" is a prompt. The
    # object is what separates them, so matching the bare verb over-flags.
    r"^\s*(?:extract|classify|rewrite|summari[sz]e|evaluate|score|audit|parse|generate)\b"
    r"[^.\n]{0,80}\b(?:the following|below|this utterance|the user'?s?|the conversation|"
    r"the response|the intent|the goal|each|as json)\b",
    r"\byou are a\b.{0,40}\b(?:engine|generator|classifier|evaluator|formatter)\b",
    r"\breturn (?:only )?(?:json|a json|valid json|the score|nothing else)\b",
    r"\brespond with (?:only|just|nothing but)\b",
    r"^\s*(?:task|instruction|context|output format)\s*:",
    r"\bdo not (?:add|include|explain|preface)\b",
)


def inspect_conversation_history(alice: Any) -> Dict[str, Any]:
    """Report whether Alice's own transcript is polluted with internal prompts.

    chat() appends to conversation_history unconditionally, including for callers
    that passed use_history=False to mark the call as machinery. Those prompts
    then come back as context on the next real turn.
    """
    engine = getattr(alice, "llm", None)
    history = list(getattr(engine, "conversation_history", []) or [])
    if not history:
        return {"turns": 0, "note": "no history recorded"}

    polluted = []
    for entry in history:
        if not isinstance(entry, dict) or entry.get("role") != "user":
            continue
        content = str(entry.get("content") or "")
        hits = [p for p in MACHINE_PROMPT_MARKERS if re.search(p, content, re.IGNORECASE | re.MULTILINE)]
        if hits:
            polluted.append(content.strip().replace("\n", " ")[:90])

    user_turns = sum(1 for e in history if isinstance(e, dict) and e.get("role") == "user")
    return {
        "turns": user_turns,
        "machine_prompts_in_history": len(polluted),
        "share": round(len(polluted) / user_turns, 2) if user_turns else 0.0,
        "examples": polluted[:5],
    }


def print_history_report(report: Dict[str, Any]) -> None:
    print("\n" + "=" * 68)
    print("WHAT ALICE THINKS WAS SAID TO HER")
    print("=" * 68)
    if not report.get("turns"):
        print(f"  {report.get('note', 'nothing recorded')}")
        return
    print(f"  user turns in history       {report['turns']}")
    print(f"  of those, machine prompts   {report['machine_prompts_in_history']} ({report['share']:.0%})")
    for example in report.get("examples", []):
        print(f"    - {example}")
    if report["machine_prompts_in_history"]:
        print("\n  She replays these as conversation on the next turn and imitates")
        print("  their register — which is how an assistant starts sounding clipped")
        print("  and instruction-shaped for no reason the user can see.")


def run_suite(
    suite: Dict[str, Any],
    *,
    model: Optional[str],
    only: Optional[List[str]],
    verbose: bool,
) -> tuple:
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
        history_report = inspect_conversation_history(alice)
    finally:
        try:
            alice.shutdown()
        except Exception:
            pass

    return results, history_report


def feel_report(results: List[TurnResult]) -> Dict[str, Any]:
    """Statistics that expose assembly rather than generation.

    No single reply proves Alice reads like a terminal. The tell is in the
    aggregate: replies that are all the same length, all end in an offer to
    help, all open the same way, and repeat verbatim when the same question is
    asked twice. These measure the shape of a whole conversation.
    """
    answered = [r for r in results if r.response.strip()]
    if not answered:
        return {"note": "no replies to analyse"}

    lengths = [r.word_count for r in answered]
    openings = [" ".join(r.response.split()[:4]).lower() for r in answered]
    endings_with_offer = [r.id for r in answered if TRAILING_OFFER.search(r.response)]
    endings_with_question = [r.id for r in answered if r.response.strip().endswith("?")]
    list_shaped = [r.id for r in answered if len(LIST_SHAPE.findall(r.response)) >= 3]
    terminal_register = [r.id for r in answered if _matches_any(TERMINAL_REGISTER, r.response)]
    restating = [r.id for r in answered if RESTATEMENT.search(r.response)]

    # Identical replies to the same prompt mean nothing was generated.
    by_prompt: Dict[str, List[str]] = {}
    for result in answered:
        by_prompt.setdefault(result.prompt.strip().lower(), []).append(result.response.strip())
    repeated = {p: v for p, v in by_prompt.items() if len(v) > 1}
    verbatim_repeats = [p for p, v in repeated.items() if len(set(v)) == 1]

    duplicate_openings = len(openings) - len(set(openings))
    spread = (max(lengths) - min(lengths)) if len(lengths) > 1 else 0

    return {
        "replies": len(answered),
        "median_words": int(statistics.median(lengths)),
        "word_range": f"{min(lengths)}-{max(lengths)}",
        "length_spread": spread,
        "uniform_length": spread <= 12 and len(lengths) > 3,
        "identical_openings": duplicate_openings,
        "ends_with_offer_to_help": f"{len(endings_with_offer)}/{len(answered)}",
        "ends_with_question": f"{len(endings_with_question)}/{len(answered)}",
        "bulleted_answers": f"{len(list_shaped)}/{len(answered)}",
        "status_report_register": f"{len(terminal_register)}/{len(answered)}",
        "restates_the_question": f"{len(restating)}/{len(answered)}",
        "verbatim_on_repeat": len(verbatim_repeats),
        "repeated_prompts_seen": len(repeated),
    }


def print_feel_report(report: Dict[str, Any]) -> None:
    print("\n" + "=" * 68)
    print("CONVERSATIONAL FEEL")
    print("=" * 68)
    for key, value in report.items():
        print(f"  {key.replace('_', ' '):<26} {value}")

    tells = []
    if report.get("uniform_length"):
        tells.append("every reply is nearly the same length — a cap or a template, not a thought")
    if int(str(report.get("identical_openings", 0))) > 1:
        tells.append("replies open with the same words — assembled from a fixed frame")
    if report.get("verbatim_on_repeat"):
        tells.append("the same question twice gave a byte-identical reply — nothing was generated")
    for metric, label in (
        ("ends_with_offer_to_help", "almost every reply ends offering further help — a script, not a conversation"),
        ("bulleted_answers", "most answers are lists — a document, not speech"),
        ("status_report_register", "replies use status-report vocabulary"),
        ("restates_the_question", "replies open by restating the question"),
    ):
        raw = str(report.get(metric, "0/1"))
        try:
            hit, total = (int(part) for part in raw.split("/"))
        except ValueError:
            continue
        if total and hit / total >= 0.5:
            tells.append(label)

    if tells:
        print("\n  What reads as a terminal:")
        for tell in tells:
            print(f"    - {tell}")
    else:
        print("\n  No aggregate tells detected.")


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
    parser.add_argument(
        "--feel",
        action="store_true",
        help="Also print aggregate conversational-feel statistics across the whole run.",
    )
    args = parser.parse_args()

    if not args.suite.exists():
        print(f"No such suite: {args.suite}", file=sys.stderr)
        return 2

    suite = json.loads(args.suite.read_text())
    only = [s.strip() for s in args.only.split(",")] if args.only else None

    results, history_report = run_suite(suite, model=args.model, only=only, verbose=args.verbose)
    if not results:
        print("No scenarios ran.", file=sys.stderr)
        return 2

    summary = summarise(results)
    print_report(results, summary)

    feel = feel_report(results) if args.feel else None
    if feel:
        print_feel_report(feel)
        print_history_report(history_report)

    if args.json:
        payload = {
            "suite": suite.get("name", args.suite.stem),
            "model": args.model or "default",
            "summary": summary,
            "feel": feel,
            "history": history_report,
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
