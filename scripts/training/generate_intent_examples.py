#!/usr/bin/env python3
"""
Intent Example Generator
========================
Generates synthetic training examples for underrepresented tool intents
and appends them to data/training/training_data.jsonl.

Usage:
    python scripts/training/generate_intent_examples.py              # fill all thin intents
    python scripts/training/generate_intent_examples.py --intent email:list
    python scripts/training/generate_intent_examples.py --min-examples 50 --count 60
    python scripts/training/generate_intent_examples.py --dry-run     # preview without saving
"""

import sys
import json
import argparse
import logging
import requests
from pathlib import Path
from datetime import datetime
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

TRAINING_FILE = Path("data/training/training_data.jsonl")

# For each intent: a short description Ollama uses to generate varied phrasings.
# "generation" is deliberately excluded — it's the catch-all we want to reduce.
INTENT_DESCRIPTIONS = {
    "email:list": "listing / checking emails (inbox, unread, new messages)",
    "email:read": "reading / opening a specific email",
    "email:search": "searching emails by sender, subject, or keyword",
    "email:compose": "composing / writing / sending a new email",
    "email:delete": "deleting / removing an email",
    "email:reply": "replying to an email",
    "notes:create": "creating / adding / writing a new note or memo",
    "notes:list": "listing / showing all notes",
    "notes:read": "reading / opening a specific note",
    "notes:search": "searching notes by keyword or title",
    "notes:delete": "deleting a note",
    "notes:query_exist": "checking whether a note exists (how many, does X note exist)",
    "weather:current": "asking about current weather conditions right now",
    "weather:forecast": "asking about future weather / forecast / tomorrow / this week",
    "time:current": "asking what time or date it is",
    "system:status": "asking about Alice's system health, status, or how she is doing",
    "music:pause": "pausing / stopping music that is playing",
    "music:play": "playing music, a song, an artist, or a playlist",
    "music:skip": "skipping to the next song / track",
    "schedule_action": "scheduling a reminder, alarm, or calendar event",
    "file_operations:create": "creating a new file or directory",
    "file_operations:delete": "deleting a file or directory",
    "file_operations:move": "moving or renaming a file",
    "code:request": "asking to read, explain, or write code",
    "conversation:help": "asking for help or what Alice can do",
    "vague_question": "vague open-ended question that needs clarification",
    "vague_request": "vague open-ended request that needs clarification",
}

# Target count per intent — intents with fewer examples than this will be filled
DEFAULT_MIN_EXAMPLES = 60
DEFAULT_GENERATE_COUNT = 60  # how many new examples to generate per thin intent


def _call_ollama(prompt: str, model: str = "llama3.1:8b", timeout: int = 60) -> str:
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.9,
            },
            timeout=timeout,
        )
        if r.status_code == 200:
            return r.json().get("response", "").strip()
    except Exception as e:
        logger.error(f"Ollama error: {e}")
    return ""


def _load_existing_counts() -> Counter:
    if not TRAINING_FILE.exists():
        return Counter()
    counts = Counter()
    for line in TRAINING_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            counts[json.loads(line).get("intent", "")] += 1
        except Exception:
            pass
    return counts


def _generate_examples(intent: str, description: str, count: int, model: str) -> list:
    """Ask Ollama for `count` diverse phrasings of `intent`. Returns list of input strings."""
    prompt = (
        f"Generate exactly {count} different ways a user might ask a personal AI assistant "
        f"to perform this action: **{description}**.\n\n"
        "Requirements:\n"
        "- Each phrase must be a natural, conversational user message (not a question about AI)\n"
        "- Use varied vocabulary, sentence structures, and formality levels\n"
        "- Include casual shorthand, full sentences, and imperative commands\n"
        "- Do NOT number the lines or add any prefix/label\n"
        "- Output ONLY the phrases, one per line, nothing else\n\n"
        "Examples of the variety expected (for a different action — do not copy these):\n"
        "  hey can you check my inbox\n"
        "  show me my emails\n"
        "  any new messages?\n"
        "  pull up my mail please\n"
    )

    raw = _call_ollama(prompt, model=model)
    if not raw:
        return []

    lines = [l.strip().lstrip("•-*0123456789.) ") for l in raw.splitlines()]
    # Filter: must be at least 5 chars, not a header or meta line
    results = [
        l
        for l in lines
        if len(l) >= 5
        and not l.lower().startswith(("here are", "sure", "of course", "note:"))
    ]
    return results[:count]


def _append_to_training_file(examples: list, intent: str, dry_run: bool = False) -> int:
    """Append examples to training_data.jsonl. Returns count written."""
    records = []
    for text in examples:
        records.append(
            {
                "user_input": text,
                "assistant_response": "",  # filled at runtime by Alice
                "context": {"intent": intent},
                "intent": intent,
                "entities": {},
                "timestamp": datetime.now().isoformat(),
                "quality_score": 0.85,
                "source": "synthetic_generated",
                "feedback": None,
            }
        )

    if dry_run:
        for r in records[:5]:
            print(f"  [DRY-RUN] {r['user_input']}")
        if len(records) > 5:
            print(f"  ... and {len(records) - 5} more")
        return len(records)

    TRAINING_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAINING_FILE, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return len(records)


def run(
    target_intents: list = None,
    min_examples: int = DEFAULT_MIN_EXAMPLES,
    generate_count: int = DEFAULT_GENERATE_COUNT,
    model: str = "llama3.1:8b",
    dry_run: bool = False,
):
    existing = _load_existing_counts()

    print(f"\n{'=' * 60}")
    print("INTENT EXAMPLE GENERATOR")
    print(f"{'=' * 60}")
    print(f"Training file: {TRAINING_FILE}")
    print(f"Min examples threshold: {min_examples}")
    print(f"Generate count per intent: {generate_count}")
    print(f"Model: {model}")
    if dry_run:
        print("MODE: DRY RUN (no files written)")
    print()

    # Determine which intents to process
    if target_intents:
        to_process = [(i, INTENT_DESCRIPTIONS.get(i, i)) for i in target_intents]
    else:
        to_process = [
            (intent, desc)
            for intent, desc in INTENT_DESCRIPTIONS.items()
            if existing.get(intent, 0) < min_examples
        ]

    if not to_process:
        print("All intents already have enough examples. Nothing to do.")
        return

    print(f"Intents to fill ({len(to_process)}):")
    for intent, _ in to_process:
        print(f"  {existing.get(intent, 0):4d} examples  →  {intent}")
    print()

    total_written = 0
    for intent, description in to_process:
        current = existing.get(intent, 0)
        needed = max(generate_count, min_examples - current)
        print(f"[{intent}]  ({current} existing, generating {needed})")

        examples = _generate_examples(intent, description, needed, model)
        if not examples:
            print(f"  WARNING: Ollama returned no examples for {intent}")
            continue

        written = _append_to_training_file(examples, intent, dry_run=dry_run)
        total_written += written
        print(f"  OK: {written} examples {'previewed' if dry_run else 'written'}")

    print()
    print(f"{'=' * 60}")
    if dry_run:
        print(f"DRY RUN complete. Would have written ~{total_written} examples.")
    else:
        print(f"Done. Total examples written: {total_written}")
        new_counts = _load_existing_counts()
        print(f"New training_data.jsonl size: {sum(new_counts.values())} records")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic intent training examples"
    )
    parser.add_argument(
        "--intent", nargs="+", help="Specific intent(s) to generate for"
    )
    parser.add_argument(
        "--min-examples",
        type=int,
        default=DEFAULT_MIN_EXAMPLES,
        help=f"Fill intents that have fewer than this many examples (default: {DEFAULT_MIN_EXAMPLES})",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_GENERATE_COUNT,
        help=f"How many examples to generate per intent (default: {DEFAULT_GENERATE_COUNT})",
    )
    parser.add_argument("--model", default="llama3.1:8b", help="Ollama model to use")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview without writing"
    )
    parser.add_argument(
        "--list", action="store_true", help="List current intent counts and exit"
    )
    args = parser.parse_args()

    if args.list:
        counts = _load_existing_counts()
        print(
            f"\nCurrent training examples per intent ({sum(counts.values())} total):\n"
        )
        for intent, count in sorted(counts.items(), key=lambda x: x[1]):
            flag = "  <-- THIN" if count < DEFAULT_MIN_EXAMPLES else ""
            print(f"  {count:4d}  {intent}{flag}")
        print()
        return

    run(
        target_intents=args.intent,
        min_examples=args.min_examples,
        generate_count=args.count,
        model=args.model,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
