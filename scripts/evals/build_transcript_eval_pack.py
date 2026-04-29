#!/usr/bin/env python3
"""Build a transcript evaluation pack from JSONL interaction logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _extract_eval_turns(rows: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    turns: List[Dict[str, Any]] = []
    for row in rows:
        user_input = str(
            row.get("user_input")
            or row.get("prompt")
            or row.get("query")
            or row.get("pattern")
            or ""
        ).strip()
        if not user_input:
            continue

        expected_route = str(
            row.get("route") or (row.get("metadata") or {}).get("route") or ""
        ).strip()
        if not expected_route:
            normalized = user_input.lower()
            if normalized.startswith("weather") or "weather" in normalized:
                expected_route = "tool"
            elif normalized.startswith("system:") or normalized.startswith("/"):
                expected_route = "tool"
            else:
                expected_route = "llm"

        expected_contains = str(
            row.get("assistant_response")
            or row.get("response")
            or row.get("ollama_phrasing")
            or ""
        ).strip()

        entry = {
            "turn": len(turns) + 1,
            "user_input": user_input,
            "expected_route": expected_route,
        }
        if expected_contains:
            entry["expected_contains"] = expected_contains[:120]

        turns.append(entry)
        if len(turns) >= limit:
            break

    return turns


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n"
    path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build transcript eval pack JSONL")
    parser.add_argument("--input", required=True, help="Input JSONL transcript path")
    parser.add_argument("--output", required=True, help="Output JSONL eval pack path")
    parser.add_argument(
        "--limit", type=int, default=50, help="Maximum turns to include"
    )
    args = parser.parse_args()

    rows = _read_jsonl(Path(args.input))
    pack = _extract_eval_turns(rows, limit=max(1, int(args.limit)))
    _write_jsonl(Path(args.output), pack)

    print(f"Wrote {len(pack)} eval turns to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
