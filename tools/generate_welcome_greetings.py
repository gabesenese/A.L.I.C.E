from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from features.welcome import _GREETING_MESSAGES
from features.welcome_validation import validate_startup_greeting


OLLAMA_URL = "http://localhost:11434/api/generate"


def _prompt_for_period(period: str, per_period: int) -> str:
    return (
        "You are generating startup banner greetings for a local AI companion/operator.\n\n"
        f"Generate short complete greetings for the period: {period}\n"
        f"Generate exactly {per_period} greetings.\n\n"
        "Style:\n"
        "- light\n- witty\n- mildly sarcastic\n- productivity-leaning\n- calm\n"
        "- not sad\n- not technical\n- not dramatic\n- not assistant-like\n\n"
        "Format:\n"
        "Each greeting must be exactly:\n"
        "Line 1: greeting with {name}\n"
        "Blank line\n"
        "Line 2: short witty/productive line\n\n"
        "Rules:\n"
        "- no questions\n- no question marks\n- no I can help\n- no how can I help\n"
        "- no assist\n- no support\n- no anything you need\n- no give me\n- no tell me\n"
        "- no point me\n- no I will\n- no sci-fi boot language\n- no fake memory\n"
        "- no project-specific claims\n- no depressing/dark jokes\n"
        "- no motivational-coach language\n- no productivity consultant jargon\n\n"
        "Good examples:\n"
        '"Morning, {name}.\\n\\nWe have a plan. Allegedly."\n'
        '"Afternoon, {name}.\\n\\nStill time for a clean win."\n'
        '"Evening, {name}.\\n\\nBack for round two."\n'
        '"Night session, {name}.\\n\\nBold choice. Let\'s make it worth it."\n'
        '"Late night, {name}.\\n\\nLet\'s keep this clever, not chaotic."\n\n'
        "Return JSON only:\n"
        "{\n"
        '  "period": "...",\n'
        '  "greetings": ["...", "..."]\n'
        "}\n"
    )


def _ollama_generate(*, model: str, prompt: str, temperature: float) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": float(temperature)},
    }
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=45) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return str(body.get("response") or "")


def _extract_json_block(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        return dict(json.loads(raw))
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        return dict(json.loads(raw[start : end + 1]))
    except Exception:
        return {}


def _validate_candidates(items: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        text = str(item or "").strip()
        reasons = validate_startup_greeting(text, require_name_placeholder=True)
        rows.append(
            {
                "text": text,
                "valid": len(reasons) == 0,
                "validation_reasons": reasons,
            }
        )
    return rows


def generate_candidates(
    *,
    model: str,
    per_period: int,
    period: str | None,
    temperature: float,
) -> dict[str, Any]:
    periods = [period] if period else list(_GREETING_MESSAGES.keys())
    output = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": model,
        "periods": {},
    }
    for p in periods:
        prompt = _prompt_for_period(p, per_period)
        raw = _ollama_generate(model=model, prompt=prompt, temperature=temperature)
        parsed = _extract_json_block(raw)
        greetings = parsed.get("greetings") if isinstance(parsed, dict) else []
        if not isinstance(greetings, list):
            greetings = []
        output["periods"][p] = _validate_candidates(
            [str(g) for g in greetings[:per_period]]
        )
    return output


def _write_json(path: Path, payload: dict[str, Any], dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] would write {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--per-period", type=int, default=25)
    parser.add_argument(
        "--output",
        default="data/generated/welcome_greeting_candidates.json",
    )
    parser.add_argument(
        "--approved-output",
        default="data/welcome_greetings.approved.json",
    )
    parser.add_argument(
        "--period",
        choices=list(_GREETING_MESSAGES.keys()),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args(argv)

    try:
        payload = generate_candidates(
            model=str(args.model),
            per_period=max(1, int(args.per_period)),
            period=args.period,
            temperature=float(args.temperature),
        )
    except URLError as exc:
        print(
            f"Ollama unavailable at {OLLAMA_URL}: {exc}. "
            "No greeting files were modified.",
            file=sys.stderr,
        )
        return 2
    except Exception as exc:
        print(f"Greeting generation failed: {exc}", file=sys.stderr)
        return 3

    _write_json(Path(args.output), payload, bool(args.dry_run))

    if not args.dry_run:
        approved_path = Path(args.approved_output)
        if not approved_path.exists():
            empty = {key: [] for key in _GREETING_MESSAGES.keys()}
            _write_json(approved_path, empty, dry_run=False)

    print("Greeting candidate generation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
